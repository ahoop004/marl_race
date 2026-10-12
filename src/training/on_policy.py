"""Synchronous TorchRL collection and the shared PPO/MAPPO update loop."""
import time
import math
import sys
from contextlib import ExitStack, contextmanager
from functools import partial, wraps

try:
    import resource
except ImportError:  # Not available on Windows.
    resource = None

import torch
from tensordict import NonTensorData, TensorDict
from torchrl.collectors import Collector
from torchrl.envs import ParallelEnv, TransformedEnv

from adapters.native_torchrl import NativeRaceTorchRLEnv
from training.contracts import validate_team_reward_composers
from training.race_reporting import RaceTrainingReports
from training.runtime import CudaCollectionProfile, preserve_random_state
from utils.torch_io import synchronize_device, resolve_collection_device
from utils.cpu_affinity import cpu_placement


class _CollectionBoundary:
    """The supported Collector interruptor protocol, evaluated after each step."""
    def __init__(self, trainer):
        self.trainer = trainer

    def collection_stopped(self):
        t = self.trainer
        return (t.budget_reached() or t._should_stop()
                or t._pending_learning_steps >= t.agent.n_steps
                or t.env.last_step is None)  # Collector has just auto-reset a race.


class OnPolicyTrainer:
    _HYBRID_COUNTERS = ("seconds", "input_speed", "target_speed", "abs_steering",
                        "blocked", "emergency_brake", "mode_switch", "no_gap",
                        "missing_scan", "invalid_state", "stopped", "braking",
                        "cold_start", "map_changed", "lidar_hits", "path_samples", "gap_candidates")
    _MPC_COUNTERS = ("seconds", "solver_seconds", "objective_seconds", "optimization_starts",
                     "nfev", "nit", "solver_failures", "outside_trajectory_evaluations",
                     "alternative_start_decisions", "alternative_starts", "braking_fallback",
                     "braking_selected", "warm_available", "warm_used", "new_plan", "cold_start",
                     "new_shoot_signatures")

    def __init__(self, task, agent, *, hooks=None, render=False, run_id="run",
                 focal_agent_id=None, algorithm=None, measure_services=False,
                 measure_stages=False, measure_mpc=False, measure_hybrid=False,
                 profiling_warmup_updates=1, measure_cuda_sync=False,
                 parallel_envs=1, envs_per_worker=1, scenario=None, scenario_dir=None, collection_device=None,
                 collection_backend="process"):
        self.task, self.agent = task, agent
        if type(parallel_envs) is not int or parallel_envs < 1:
            raise ValueError("parallel_envs must be a positive integer")
        if type(envs_per_worker) is not int or envs_per_worker < 1:
            raise ValueError("envs_per_worker must be a positive integer")
        self.envs_per_worker = envs_per_worker
        self.total_envs = parallel_envs * envs_per_worker
        if not isinstance(collection_backend, str) or collection_backend not in {"process", "centralized"}:
            raise ValueError("collection_backend must be process or centralized")
        self.collection_backend = collection_backend
        self.centralized_collection = collection_backend == "centralized"
        self.process_collection = self.total_envs > 1 or self.centralized_collection
        if self.process_collection:
            if scenario is None:
                raise ValueError("Process collection requires the task's scenario configuration")
            if render:
                raise ValueError("Process collection requires headless environments")
        self.parallel_envs, self.scenario = parallel_envs, scenario
        self._process_placement = cpu_placement()
        self._process_placement.pop("cpu_topology")
        self._collection_batches = self._last_collection_batch_frames = 0
        self.collection_device = resolve_collection_device(
            collection_device, agent.device, parallel_envs, envs_per_worker, collection_backend)
        self.scenario_dir = str(scenario_dir) if scenario_dir is not None else None
        self._worker_policy_starts = {}
        self.algorithm = algorithm or ("mappo" if hasattr(agent, "global_state_dim") else "ppo")
        if self.algorithm not in {"ppo", "mappo"}:
            raise ValueError(f"Unsupported on-policy algorithm: {self.algorithm!r}")
        self.hooks, self.render, self.run_id = hooks or [], render, run_id
        if type(measure_services) is not bool:
            raise ValueError("measure_services must be a boolean")
        self.measure_services = measure_services
        self.service_seconds = dict(logging=0.0, evaluation=0.0, checkpoint=0.0)
        if type(measure_stages) is not bool:
            raise ValueError("measure_stages must be a boolean")
        if type(profiling_warmup_updates) is not int or profiling_warmup_updates < 0:
            raise ValueError("profiling_warmup_updates must be a nonnegative integer")
        if type(measure_mpc) is not bool:
            raise ValueError("measure_mpc must be a boolean")
        self.measure_mpc = measure_mpc
        self.mpc_totals = {}
        self.mpc_environment_decisions = {}
        if type(measure_hybrid) is not bool:
            raise ValueError("measure_hybrid must be a boolean")
        self.measure_hybrid = measure_hybrid
        self.hybrid_totals = {}
        self.hybrid_environment_decisions = {}
        self.measure_stages = measure_stages
        if type(measure_cuda_sync) is not bool:
            raise ValueError("measure_cuda_sync must be a boolean")
        self.measure_cuda_sync = measure_cuda_sync
        self.profiling_warmup_updates = profiling_warmup_updates
        self.stage_seconds = {phase: {} for phase in ("initialization", "warmup", "steady")}
        self.stage_calls = {phase: {} for phase in self.stage_seconds}
        self.scope_seconds = {phase: {} for phase in self.stage_seconds}
        self.process_cpu_seconds = {phase: 0.0 for phase in self.stage_seconds}
        self._process_phase_started = None
        self._peak_rss_bytes = None
        self._timing_phase = "initialization"
        self.initialization_seconds = 0.0
        self._run_started = self._run_finished = self._steady_started = None
        self._steady_finished = None
        self._rate_baseline = self._steady_baseline = None
        self.boundary_seconds = {phase: {} for phase in self.stage_seconds}
        self.worker_stage_seconds = {phase: {} for phase in self.stage_seconds}
        self.worker_stage_calls = {phase: {} for phase in self.stage_seconds}
        self.worker_scope_seconds = {phase: {} for phase in self.stage_seconds}
        self.worker_metrics = {}
        self.worker_batch_metrics = {}
        self.inference_profile = {phase: {} for phase in self.stage_seconds}
        self.cuda_copy_seconds = 0.
        self.cuda_copy_calls = 0
        self.cuda_upload_seconds = 0.
        self.cuda_upload_calls = 0
        self.trainable_ids = list(task.possible_agents)
        self.focal_id = focal_agent_id or self.trainable_ids[0]
        if self.focal_id not in self.trainable_ids:
            raise ValueError("Focal agent must be a policy agent")
        if self.algorithm == "ppo":
            if len(self.trainable_ids) != 1 or task.team_reward_agent_id is not None:
                raise ValueError("PPO requires exactly one learner and individual rewards")
        else:
            if list(agent.agent_ids) != self.trainable_ids:
                raise ValueError("Task policy agents must exactly match MAPPO agent ID order")
            validate_team_reward_composers(
                task.reward_composers, trainable_ids=self.trainable_ids,
                opponent_ids=list(task.fixed_controllers), reward_mode=agent.reward_mode,
                critic_mode=agent.critic_mode, team_return_mode=agent.team_return_mode,
                action_repeat=task.action_repeat)
            if task.action_repeat != 1:
                raise ValueError("Joint team returns require action_repeat=1")
        self._environment_steps = self._physics_steps = self._agent_steps = self._updates = 0
        # n_steps targets global active actor samples. Serial MAPPO overshoots
        # by at most learners - 1; processes stop at a physical batch boundary.
        self._pending_learning_steps = 0
        self.total_steps = self.n_episodes = None
        self.env = None
        self.reports = RaceTrainingReports(self)
        self._optimizer_steps = 0
        self._resume_state = None
        self.checkpoint_ready = False
        for hook in self.hooks:
            bind = getattr(hook, "bind_trainer", None)
            if bind:
                bind(self)

    def measure_service(self, category, callback, *args, **kwargs):
        if not self.measure_services or category is None:
            return callback(*args, **kwargs)
        previous = sum(self.service_seconds.values())
        started = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            nested = sum(self.service_seconds.values()) - previous
            self.service_seconds[category] += max(0.0, time.perf_counter() - started - nested)

    def _stage_total(self):
        return sum(sum(values.values()) for values in self.stage_seconds.values())

    def measure_stage(self, category, callback, *args, device=None, **kwargs):
        if not self.measure_stages:
            return callback(*args, **kwargs)
        with self.measure_region(category, device=device):
            return callback(*args, **kwargs)

    @contextmanager
    def measure_region(self, category, *, device=None):
        """Exclusive wall time; extra CUDA synchronization is separately opt-in."""
        if not self.measure_stages:
            yield
            return
        phase = self._timing_phase
        previous = self._stage_total() + sum(self.service_seconds.values())
        started = time.perf_counter()
        sync = (self.measure_cuda_sync and device is not None
                and torch.device(device).type == "cuda")
        try:
            if sync:
                self.measure_stage("cuda_synchronization", synchronize_device, device)
            yield
        finally:
            if sync:
                self.measure_stage("cuda_synchronization", synchronize_device, device)
            elapsed = time.perf_counter() - started
            nested = self._stage_total() + sum(self.service_seconds.values()) - previous
            values, calls = self.stage_seconds[phase], self.stage_calls[phase]
            values[category] = values.get(category, 0.0) + max(0.0, elapsed - nested)
            calls[category] = calls.get(category, 0) + 1
            # Inclusive envelopes are reported separately and never summed with stages.
            scope = {"environment_adapter": "environment_step", "physics_overhead": "physics_step"}.get(category)
            if scope:
                totals = self.scope_seconds[phase]
                totals[scope] = totals.get(scope, 0.0) + elapsed

    def _instrument_stages(self, resources):
        # Only measured runs install wrappers. Restore existing instance methods
        # when collection ends; no global patches or private Collector state.
        def instrument(owner, name, category, device=None):
            original = getattr(owner, name, None)
            if original is None:
                return
            own_value = vars(owner).get(name)
            had_value = name in vars(owner)
            @wraps(original)
            def timed(*args, **kwargs):
                return self._profile_learning(category, original, *args, device=device, **kwargs)
            setattr(owner, name, timed)
            def restore():
                if had_value:
                    setattr(owner, name, own_value)
                else:
                    delattr(owner, name)
            resources.callback(restore)

        instrument(self.agent.gae, "forward", "gae", self.agent.device)
        self.agent.profile_call = self._profile_learning
        resources.callback(setattr, self.agent, "profile_call", None)
        if self.process_collection or not self.measure_stages:
            # Inference and environments live in workers. Never serialize a
            # policy wrapper closing over the parent trainer and its hooks.
            return
        # Worker policy-version transforms dispatch to the native methods;
        # the direct collector calls the public step/reset entrypoints.
        worker_local = self.agent.collection_policy is None
        instrument(self.env, "_step" if worker_local else "step", "environment_adapter")
        instrument(self.env, "_reset" if worker_local else "reset", "environment_reset")
        instrument(self.env, "_output", "tensor_construction")
        instrument(self.task, "step", "race_bookkeeping_reward")
        instrument(self.task.env, "step", "race_bookkeeping_reward")
        if self.collection_device == self.agent.device:
            instrument(self.agent.collection_policy, "forward", "policy_inference", self.collection_device)
        for composer in self.task.obs_composers.values():
            instrument(composer, "wrap", "observation_state")
        for composer in self.task.reward_composers.values():
            instrument(composer, "compute", "race_bookkeeping_reward")
        instrument(self.task, "_make_snapshot", "observation_state")
        for controller in self.task.fixed_controllers.values():
            instrument(controller, "act", "fixed_controller")
        for name in ("_split_obs", "_update_centerline_observation_facts", "_attach_central_state", "get_global_state"):
            instrument(self.task.env, name, "observation_state")
        simulator = getattr(self.task.env, "sim", None)
        if simulator is not None:
            instrument(simulator, "step", "physics_overhead")
            instrument(simulator, "refresh_scans", "lidar")
            instrument(simulator.scan_simulator, "scan", "lidar")
            for vehicle in simulator.agents:
                instrument(vehicle, "update_pose", "physics_dynamics")
                instrument(vehicle, "compute_scan", "lidar")
                instrument(vehicle, "check_ttc", "physics_collisions")
                instrument(vehicle, "ray_cast_agents", "lidar")
            original = getattr(simulator, "profile_region", None)
            simulator.profile_region = self.measure_region
            resources.callback(setattr, simulator, "profile_region", original)
        original = getattr(self.task, "profile_region", None)
        self.task.profile_region = self.measure_region
        resources.callback(setattr, self.task, "profile_region", original)
        self.env.profile_call = self.measure_stage
        resources.callback(setattr, self.env, "profile_call", None)

    def _profile_learning(self, category, callback, *args, device=None, **kwargs):
        # Coarse learning timers stay enabled in benchmarks. Detailed CUDA
        # completion timings remain opt-in through measure_cuda_sync.
        if category not in {"rollout_device_transfer", "gae", "optimization"}:
            return self.measure_stage(category, callback, *args, device=device, **kwargs)
        started = time.perf_counter()
        try:
            return self.measure_stage(category, callback, *args, device=device, **kwargs)
        finally:
            self._record_boundary(category, started)

    def _instrument_mpc(self, resources):
        from agents.mpc.racing import RacingMPCAgent
        from wrappers.actions.composer import WheelReferenceAdapter

        self._mpc_ids = []
        for aid, adapter in self.task.fixed_controllers.items():
            controller = adapter.controller if isinstance(adapter, WheelReferenceAdapter) else adapter
            if not isinstance(controller, RacingMPCAgent):
                continue
            self._mpc_ids.append(aid)
            self.mpc_totals.setdefault("agent", {}).setdefault(aid, {})
            original = controller.profile_mpc
            controller.profile_mpc = self._record_mpc
            resources.callback(setattr, controller, "profile_mpc", original)

    def _record_mpc(self, record):
        # Read the same live physical lifecycle used by RaceTask.step. Retired
        # wrecks can still be nearby traffic; they are not active vehicles.
        active = self.task.env.agents
        context = dict(phase=self._timing_phase, agent=record["agent_id"],
                       active_vehicles=len(active),
                       active_learners=sum(aid in active for aid in self.trainable_ids),
                       active_mpc=sum(aid in active for aid in self._mpc_ids),
                       traffic=record["traffic_count"], starts=record["optimization_starts"],
                       warm_used=record["warm_used"], cold_start=record["cold_start"],
                       new_shoot_signatures=int(record["new_shoot_signatures"] > 0))
        groups = [("all", "all"), *context.items(),
                  ("agent_phase", f"{context['agent']}/{context['phase']}")]
        for dimension, key in groups:
            bucket = self.mpc_totals.setdefault(dimension, {}).setdefault(str(key), {})
            bucket["calls"] = bucket.get("calls", 0)+1
            for name in self._MPC_COUNTERS:
                bucket[name] = bucket.get(name, 0)+record[name]
            for name in ("stages", "solver_statuses"):
                totals = bucket.setdefault(name, {})
                for stage, value in record[name].items():
                    totals[stage] = totals.get(stage, 0)+value

    @classmethod
    def _mpc_statistics(cls, bucket):
        result = {name: bucket.get(name, 0) for name in cls._MPC_COUNTERS}
        result["solver_statuses"] = dict(bucket.get("solver_statuses", {}))
        calls, seconds = bucket.get("calls", 0), bucket.get("seconds", 0.0)
        starts, nfev = bucket.get("optimization_starts", 0), bucket.get("nfev", 0)
        result.update(calls=calls, seconds=seconds,
                      mean_ms_per_call=1000*seconds/calls if calls else 0.0,
                      mean_nfev_per_optimization=nfev/starts if starts else 0.0,
                      mean_nit_per_optimization=bucket.get("nit", 0)/starts if starts else 0.0,
                      solver_failure_rate=bucket.get("solver_failures", 0)/starts if starts else 0.0,
                      mean_solver_ms=1000*bucket.get("solver_seconds", 0)/starts if starts else 0.0,
                      mean_objective_ms=1000*bucket.get("objective_seconds", 0)/nfev if nfev else 0.0)
        for name in ("braking_fallback", "braking_selected", "alternative_start_decisions", "warm_used"):
            result[f"{name}_rate"] = bucket.get(name, 0)/calls if calls else 0.0
        result["stages"] = {name: dict(seconds=value,
            mean_ms_per_call=1000*value/calls if calls else 0.0,
            percent_mpc=100*value/seconds if seconds else 0.0)
            for name, value in bucket.get("stages", {}).items()}
        return result

    def mpc_summary(self):
        if not self.measure_mpc:
            return {}
        groups = {dimension: {key: self._mpc_statistics(bucket) for key, bucket in values.items()}
                  for dimension, values in self.mpc_totals.items()}
        overall = groups.pop("all", {}).get("all", self._mpc_statistics({}))
        for dimension, keys in (("phase", self.stage_seconds), ("active_vehicles", range(1, 5))):
            values = groups.setdefault(dimension, {})
            for key in keys:
                values.setdefault(str(key), self._mpc_statistics({}))
        decisions = sum(self.mpc_environment_decisions.values())
        now = self._run_finished or time.perf_counter()
        elapsed = max(now-self._run_started, 0.0) if self._run_started is not None else 0.0
        overall.update(environment_decisions=decisions,
                       ms_per_environment_decision=1000*overall["seconds"]/decisions if decisions else 0.0,
                       end_to_end_seconds=elapsed,
                       percent_end_to_end=100*overall["seconds"]/elapsed if elapsed else 0.0)
        for phase, values in groups["phase"].items():
            count = self.mpc_environment_decisions.get(phase, 0)
            values["environment_decisions"] = count
            values["ms_per_environment_decision"] = 1000*values["seconds"]/count if count else 0.0
        return dict(overall=overall, **groups, definitions={
            "scope": "Training collection only; totals reset at each train() call and accumulate across episodes.",
            "seconds": "Wall time inside RacingMPCAgent.act, excluding adapter and post-call aggregation.",
            "stages": "Exclusive seconds; scipy_optimization excludes objective_evaluation. other includes bookkeeping.",
            "solver_seconds": "Inclusive minimize wall time; diagnostic only, do not add to exclusive stages.",
            "nfev_nit": "Summed SciPy OptimizeResult metadata; means divide by optimization_starts.",
            "environment_decisions": "All training joint decisions, including decisions with no active MPC opponents.",
            "percent_end_to_end": "MPC seconds / trainer wall time, including collection, updates, and hooks; excludes application setup.",
            "activity": "RaceEnv.agents; nearby traffic can include retired stationary vehicles.",
            "braking": "fallback means no feasible candidate; selected also counts feasible braking candidates.",
            "phase": "Collector initialization, then profiling_warmup_updates completed updates, then steady.",
            "cold_start": "First MPC decision of each episode; map/JIT work is included in its phase.",
            "new_shoot_signatures": "Calls adding Numba shooting signatures, including cache loads; not isolated JIT compile time.",
            "groups": "Each grouping is an independent partition; do not sum different dimensions."})

    def mpc_metrics(self):
        if not self.measure_mpc:
            return {}
        summary = self.mpc_summary()
        result = {}
        for prefix, values in [("all", summary["overall"]),
                               *[(f"agent/{key}", value) for key, value in summary.get("agent", {}).items()],
                               *[(f"phase/{key}", value) for key, value in summary["phase"].items()]]:
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    result[f"perf/mpc/{prefix}/{key}"] = value
        return result

    def _instrument_hybrid(self, resources):
        from agents.fixed.hybrid import HybridPPFTGAgent
        from wrappers.actions.composer import WheelReferenceAdapter

        self._hybrid_ids = []
        for aid, adapter in self.task.fixed_controllers.items():
            controller = adapter.controller if isinstance(adapter, WheelReferenceAdapter) else adapter
            if not isinstance(controller, HybridPPFTGAgent):
                continue
            self._hybrid_ids.append(aid)
            self.hybrid_totals.setdefault("agent", {}).setdefault(aid, {})
            original = controller.profile_hybrid
            controller.profile_hybrid = self._record_hybrid
            resources.callback(setattr, controller, "profile_hybrid", original)

    def _record_hybrid(self, record):
        active = self.task.env.agents
        context = dict(phase=self._timing_phase, agent=record["agent_id"],
                       active_vehicles=len(active),
                       active_learners=sum(aid in active for aid in self.trainable_ids),
                       active_hybrid=sum(aid in active for aid in self._hybrid_ids),
                       traffic=record["traffic_count"], mode=record["mode"],
                       blocked=record["blocked"], emergency_brake=record["emergency_brake"],
                       invalid_state=record["invalid_state"], cold_start=record["cold_start"],
                       map_changed=record["map_changed"])
        groups = [("all", "all"), *context.items(),
                  ("agent_phase", f"{context['agent']}/{context['phase']}")]
        for dimension, key in groups:
            bucket = self.hybrid_totals.setdefault(dimension, {}).setdefault(str(key), {})
            bucket["calls"] = bucket.get("calls", 0) + 1
            for name in self._HYBRID_COUNTERS:
                bucket[name] = bucket.get(name, 0) + record[name]
            bucket["seconds_squared"] = bucket.get("seconds_squared", 0.) + record["seconds"] ** 2
            bucket["max_seconds"] = max(bucket.get("max_seconds", 0.), record["seconds"])
            for name in ("stages", "stage_calls"):
                totals = bucket.setdefault(name, {})
                for stage, value in record[name].items():
                    totals[stage] = totals.get(stage, 0) + value

    @classmethod
    def _hybrid_statistics(cls, bucket):
        mean_counters = {"input_speed", "target_speed", "abs_steering", "lidar_hits", "path_samples", "gap_candidates"}
        result = {name: bucket.get(name, 0) for name in cls._HYBRID_COUNTERS if name not in mean_counters}
        calls, seconds = bucket.get("calls", 0), bucket.get("seconds", 0.)
        mean = seconds / calls if calls else 0.
        result.update(calls=calls, mean_ms_per_call=1000 * mean,
                      max_ms_per_call=1000 * bucket.get("max_seconds", 0.),
                      std_ms_per_call=1000 * math.sqrt(max(0., bucket.get("seconds_squared", 0.) / calls - mean ** 2)) if calls else 0.)
        for name in cls._HYBRID_COUNTERS:
            if name == "seconds":
                continue
            key = f"mean_{name}" if name in mean_counters else f"{name}_rate"
            result[key] = bucket.get(name, 0) / calls if calls else 0.
        result["stages"] = {}
        for name in ("map_preparation", "pure_pursuit", "speed_planning", "lidar_preprocessing",
                     "traffic_prediction", "path_checks", "gap_selection", "smoothing", "other"):
            value = bucket.get("stages", {}).get(name, 0.)
            invocations = bucket.get("stage_calls", {}).get(name, 0)
            result["stages"][name] = dict(seconds=value, calls=invocations,
                mean_ms_per_call=1000 * value / calls if calls else 0.,
                mean_ms_per_invocation=1000 * value / invocations if invocations else 0.,
                percent_hybrid=100 * value / seconds if seconds else 0.)
        return result

    def hybrid_summary(self):
        if not self.measure_hybrid:
            return {}
        groups = {dimension: {key: self._hybrid_statistics(bucket) for key, bucket in values.items()}
                  for dimension, values in self.hybrid_totals.items()}
        overall = groups.pop("all", {}).get("all", self._hybrid_statistics({}))
        for dimension, keys in (("phase", self.stage_seconds), ("active_vehicles", range(1, 5)),
                                ("mode", ("pp", "ftg", "invalid"))):
            values = groups.setdefault(dimension, {})
            for key in keys:
                values.setdefault(str(key), self._hybrid_statistics({}))
        decisions = sum(self.hybrid_environment_decisions.values())
        now = self._run_finished or time.perf_counter()
        elapsed = max(now - self._run_started, 0.) if self._run_started is not None else 0.
        overall.update(environment_decisions=decisions,
                       ms_per_environment_decision=1000 * overall["seconds"] / decisions if decisions else 0.,
                       end_to_end_seconds=elapsed,
                       percent_end_to_end=100 * overall["seconds"] / elapsed if elapsed else 0.)
        for phase, values in groups["phase"].items():
            count = self.hybrid_environment_decisions.get(phase, 0)
            values["environment_decisions"] = count
            values["ms_per_environment_decision"] = 1000 * values["seconds"] / count if count else 0.
        for values in groups["mode"].values():
            values["fraction"] = values["calls"] / overall["calls"] if overall["calls"] else 0.
        return dict(overall=overall, **groups, definitions={
            "scope": "Training collection only; totals reset at each train() call and accumulate across episodes.",
            "seconds": "Wall time inside HybridPPFTGAgent.act, excluding adapter conversion and post-call aggregation.",
            "stages": "Exclusive sequential seconds; other covers bookkeeping and early-return work. Pure Pursuit is computed in FTG calls too.",
            "stage_means": "mean_ms_per_call divides by all controller calls; mean_ms_per_invocation divides by that stage's calls.",
            "environment_decisions": "All training joint decisions, including decisions with no active hybrid opponents.",
            "percent_end_to_end": "Hybrid seconds / trainer wall time including updates and hooks, excluding application setup.",
            "mode": "Final decision mode; invalid labels invalid pose, velocity, yaw, steering, or grip. FTG includes hysteresis and missing-scan stops.",
            "braking": "Commanded target speed below measured forward speed by more than 0.001 m/s; stopped means target <= 0.000001 m/s.",
            "mode_switch": "PP/FTG transitions within valid decisions; episode/map resets are not switches.",
            "no_gap": "FTG search returned no gap; missing scans and emergency traffic braking have separate counters.",
            "phase": "Collector initialization, then profiling_warmup_updates completed updates, then steady.",
            "cold_start": "First decision after episode/map reset; map preparation and any JIT/cache work remain in the calling stage.",
            "traffic": "Forward nearby cars used by the short traffic envelope, including retired stationary wrecks; active counts use RaceEnv.agents.",
            "dispersion": "Population standard deviation and maximum per-call time; no per-decision samples retained.",
            "work": "Mean lidar_hits counts occupied scan returns; path_samples counts centerline preview points; gap_candidates counts tested steering arcs across gap tiers.",
            "commands": "Mean input/target speed in m/s and absolute physical steering in radians, before adapter conversion.",
            "groups": "Each grouping is an independent partition; do not sum different dimensions."})

    def hybrid_metrics(self):
        if not self.measure_hybrid:
            return {}
        summary = self.hybrid_summary()
        result = {}
        for prefix, values in [("all", summary["overall"]),
                               *[(f"agent/{key}", value) for key, value in summary.get("agent", {}).items()],
                               *[(f"phase/{key}", value) for key, value in summary["phase"].items()],
                               *[(f"mode/{key}", value) for key, value in summary["mode"].items()]]:
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    result[f"perf/hybrid/{prefix}/{key}"] = value
            for stage, fields in values["stages"].items():
                for key, value in fields.items():
                    result[f"perf/hybrid/{prefix}/stages/{stage}/{key}"] = value
        return result

    def _rate_counters(self):
        return (self._environment_steps, self._physics_steps, self._agent_steps, self._optimizer_steps)

    def _finish_process_phase(self):
        if self._process_phase_started is not None:
            self.process_cpu_seconds[self._timing_phase] += time.process_time() - self._process_phase_started
            self._process_phase_started = None

    def _set_timing_phase(self, phase):
        self._finish_process_phase()
        self._timing_phase = phase
        self._process_phase_started = time.process_time()

    def _record_boundary(self, name, started):
        values = self.boundary_seconds[self._timing_phase]
        values[name] = values.get(name, 0.) + time.perf_counter() - started

    @staticmethod
    def _merge_profile(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                OnPolicyTrainer._merge_profile(target.setdefault(key, {}), value)
            elif key == "max_seconds":
                target[key] = max(target.get(key, 0.), value)
            else:
                target[key] = target.get(key, 0) + value

    def _receive_worker_profile(self, worker, packet):
        self.worker_metrics[worker] = packet["process"]
        self._merge_profile(self.inference_profile[self._timing_phase], packet.get("inference", {}))
        for name in ("stage_seconds", "stage_calls", "scope_seconds"):
            target = getattr(self, "worker_" + name)
            for phase, values in packet.get(name, {}).items():
                phase = "initialization" if phase == "initialization" else self._timing_phase
                self._merge_profile(target[phase], values)
        for kind in ("mpc", "hybrid"):
            groups = packet.get(kind, {})
            for dimension in ("phase", "agent_phase"):
                groups[dimension] = {
                    (self._timing_phase if dimension == "phase" else key.rsplit("/", 1)[0] + "/" + self._timing_phase): values
                    for key, values in groups.get(dimension, {}).items()}
            self._merge_profile(getattr(self, kind + "_totals"), groups)

    def _sample_peak_rss(self):
        if resource is not None:
            # ru_maxrss is bytes on macOS and KiB on other supported Unix hosts.
            self._peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024)

    def performance_metrics(self):
        if self._run_started is None:
            return {}
        now = self._run_finished or time.perf_counter()
        elapsed = max(now - self._run_started, 1e-9)
        names = ("environment_steps", "physics_steps", "learner_samples", "optimizer_steps")
        result = {f"perf/{name}_per_second": (value - base) / elapsed
                  for name, value, base in zip(names, self._rate_counters(), self._rate_baseline)}
        result["perf/elapsed_seconds"] = elapsed
        result["perf/end_to_end_env_steps_per_second"] = result["perf/environment_steps_per_second"]
        result.update({"perf/collection_device": str(self.collection_device),
                       "perf/collection_backend": self.collection_backend,
                       "perf/learner_device": str(self.agent.device),
                       "perf/worker_count": self.parallel_envs,
                       "perf/envs_per_worker": self.envs_per_worker,
                       "perf/environment_slots": self.total_envs,
                       "perf/worker_wait_overlaps_collection": True,
                       "perf/worker_collection_includes_central_inference": self.centralized_collection,
                       "perf/learning_cuda_completion_timing": self.measure_stages and self.measure_cuda_sync,
                       "perf/action_transfer_in_collection": self.collection_device.type == "cuda"})
        if self.measure_stages and self.measure_cuda_sync and self.collection_device.type == "cuda":
            result["perf/cuda_collection/device_to_cpu_seconds"] = self.cuda_copy_seconds + sum(
                values.get("collection_device_to_cpu_seconds", 0.) for values in self.worker_metrics.values())
            result["perf/cuda_collection/device_to_cpu_calls"] = self.cuda_copy_calls + sum(
                values.get("collection_device_to_cpu_calls", 0) for values in self.worker_metrics.values())
            result["perf/cuda_collection/cpu_to_device_seconds"] = self.cuda_upload_seconds + sum(
                values.get("collection_cpu_to_device_seconds", 0.) for values in self.worker_metrics.values())
            result["perf/cuda_collection/cpu_to_device_calls"] = self.cuda_upload_calls + sum(
                values.get("collection_cpu_to_device_calls", 0) for values in self.worker_metrics.values())
        cuda_device = next((device for device in (self.agent.device, self.collection_device)
                            if torch.device(device).type == "cuda"), None)
        if cuda_device is not None:
            result["perf/cuda_device_name"] = torch.cuda.get_device_name(cuda_device)
            result["perf/cuda_device_index"] = torch.device(cuda_device).index or 0
            result["perf/cuda_parent_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(cuda_device)
            result["perf/cuda_parent_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(cuda_device)
        result.update({f"perf/boundaries/{phase}/{name}_seconds": seconds
            for phase, values in self.boundary_seconds.items() for name, seconds in values.items()})
        for worker, values in self.worker_metrics.items():
            result.update({f"perf/workers/{worker}/{name}": value for name, value in values.items()})
        result.update(self.worker_batch_metrics)
        if self.measure_stages:
            result["perf/inference/cuda_completion_timing"] = self.measure_cuda_sync
            result["perf/inference/worker_seconds_overlap"] = not self.centralized_collection
            for phase, values in self.inference_profile.items():
                if not values.get("calls"):
                    continue
                prefix = f"perf/inference/{phase}/"
                result.update({prefix + name: value for name, value in values.items()})
                result.update({prefix + "mean_environment_batch": values["environment_rows"] / values["calls"],
                               prefix + "mean_agent_batch": values["agent_rows"] / values["calls"],
                               prefix + "mean_latency_ms": 1000 * values["seconds"] / values["calls"]})
        result.update({f"perf/worker_stages/{phase}/{name}_seconds": seconds
            for phase, values in self.worker_stage_seconds.items() for name, seconds in values.items()})
        result.update({f"perf/worker_stages/{phase}/{name}_calls": calls
            for phase, values in self.worker_stage_calls.items() for name, calls in values.items()})
        result.update({f"perf/worker_scopes/{phase}/{name}_seconds": seconds
            for phase, values in self.worker_scope_seconds.items() for name, seconds in values.items()})
        cpu = dict(self.process_cpu_seconds)
        if self._process_phase_started is not None:
            cpu[self._timing_phase] += time.process_time() - self._process_phase_started
        result.update({f"perf/process/{phase}_cpu_seconds": seconds for phase, seconds in cpu.items()})
        result["perf/process/cpu_seconds"] = sum(cpu.values())
        result["perf/process/cpu_percent"] = 100 * sum(cpu.values()) / elapsed
        if self._run_finished is None:
            self._sample_peak_rss()
        if self._peak_rss_bytes is not None:
            result["perf/process/peak_rss_bytes"] = self._peak_rss_bytes
        result.update({f"perf/process/{name}": value for name, value in self._process_placement.items()})
        result["perf/collection_batches"] = self._collection_batches
        result["perf/last_collection_batch_frames"] = self._last_collection_batch_frames
        if self._steady_started is not None:
            duration = max((self._steady_finished or now) - self._steady_started, 1e-9)
            result["perf/steady_elapsed_seconds"] = duration
            result.update({f"perf/steady_{name}": value - base
                for name, value, base in zip(names, self._rate_counters(), self._steady_baseline)})
            result.update({f"perf/steady_{name}_per_second": (value - base) / duration
                for name, value, base in zip(names, self._rate_counters(), self._steady_baseline)})
        if self.measure_stages:
            result.update({f"perf/stages/{phase}/{name}_seconds": seconds
                for phase, values in self.stage_seconds.items() for name, seconds in values.items()})
            result.update({f"perf/stages/{phase}/{name}_calls": calls
                for phase, values in self.stage_calls.items() for name, calls in values.items()})
            result.update({f"perf/scopes/{phase}/{name}_seconds": seconds
                for phase, values in self.scope_seconds.items() for name, seconds in values.items()})
            result["perf/measure_cuda_sync"] = self.measure_cuda_sync
            result["perf/initialization_seconds"] = self.initialization_seconds + sum(self.stage_seconds["initialization"].values())
            result["perf/warmup_seconds"] = sum(self.stage_seconds["warmup"].values())
        result.update(self.mpc_metrics())
        result.update(self.hybrid_metrics())
        return result

    def call_hook(self, hook, method, *args):
        callback = getattr(hook, method, None)
        if callback:
            return self.measure_service(getattr(hook, "service_name", None), callback, *args)

    def service_metrics(self):
        return ({f"perf/{name}_seconds": seconds for name, seconds in self.service_seconds.items()}
                if self.measure_services else {})

    def checkpoint_progress(self):
        return {"environment_steps": self._environment_steps, "agent_steps": self._agent_steps,
                "physics_steps": self._physics_steps, "completed_episodes": self.reports.completed,
                "updates": self._updates, "optimizer_steps": self._optimizer_steps,
                "next_episode": self.reports.next_episode}

    def counter_metrics(self):
        metrics = {"train/environment_steps": self._environment_steps,
                "train/physics_steps": self._physics_steps,
                "train/agent_steps": self._agent_steps,
                "train/completed_episodes": self.reports.completed,
                "train/updates": self._updates,
                "train/optimizer_steps_total": self._optimizer_steps}
        if self.process_collection:
            metrics.update({
                "train/budget_step_overshoot": max(0, self._environment_steps - self.total_steps) if self.total_steps is not None else 0,
                "train/budget_episode_overshoot": max(0, self.reports.completed - self.n_episodes) if self.total_steps is None and self.n_episodes is not None else 0,
            })
        return metrics

    def checkpoint_state(self):
        """Recovery at a completed update; the next collection starts a new race.

        Simulator/controller state and partial episodes are deliberately not
        serialized. Map/spawn/friction generators restart at recovery_seed.
        This is deterministic recovery, not an exact next-transition resume.
        """
        from agents.common.checkpoints import RESUME_BOUNDARY, capture_rng, optimization_contract
        if not self.checkpoint_ready or getattr(self.agent, "_pending_batches", []):
            raise ValueError("Resumable checkpoints require a completed update with no pending trajectories")
        completed, total = ((self._environment_steps, self.total_steps) if self.total_steps is not None
                            else (self.reports.completed, self.n_episodes))
        return {"boundary": RESUME_BOUNDARY, "optimizer": self.agent.optimizer.state_dict(),
                "optimization_contract": optimization_contract(self.agent),
                "schedule_progress": completed / max(total, 1), "progress": self.checkpoint_progress(),
                "budget": {"total_steps": self.total_steps, "n_episodes": self.n_episodes},
                "rng": capture_rng(), "recovery_seed": getattr(self.task.env, "seed", None) or 0,
                "hooks": {type(hook).__name__: hook.state_dict() for hook in self.hooks
                          if hasattr(hook, "state_dict")},
                "recovery": {"discard_partial_episode": True, "environment_schedules": "restart_at_seed"}}

    def restore_training_state(self, state):
        self._resume_state = state
        progress = state["progress"]
        self._environment_steps, self._agent_steps, self._physics_steps = (
            progress["environment_steps"], progress["agent_steps"], progress["physics_steps"])
        self._updates, self._optimizer_steps = progress["updates"], progress["optimizer_steps"]
        self.reports.completed = progress["completed_episodes"]
        self.reports.next_episode = progress["next_episode"]
        for hook in self.hooks:
            if hasattr(hook, "load_state_dict"):
                if type(hook).__name__ not in state["hooks"]:
                    raise ValueError(f"Resume checkpoint missing state for {type(hook).__name__}")
                hook._environment_steps, hook._policy_version = self._environment_steps, self._updates
                hook.load_state_dict(state["hooks"][type(hook).__name__])

    @property
    def collected_steps(self):
        return self._environment_steps

    def _should_stop(self):
        return bool(getattr(self.agent, "should_stop", False)) or any(
            getattr(hook, "should_stop", False) for hook in self.hooks)

    def budget_reached(self):
        if self.total_steps is not None:
            return self._environment_steps >= self.total_steps
        return self.n_episodes is not None and self.reports.completed >= self.n_episodes

    def _set_progress(self):
        setter = getattr(self.agent, "set_training_progress", None)
        if setter:
            completed, total = ((self._environment_steps, self.total_steps) if self.total_steps is not None
                                else (self.reports.completed, self.n_episodes))
            setter(completed / max(total, 1))

    def _process_factories(self):
        from training.process_collection import RaceEnvFactory
        seed = (self._resume_state["recovery_seed"] if self._resume_state is not None
                else self.scenario["experiment"].get("seed"))
        # A missing seed still gets a reproducible worker schedule for this run.
        seed = 0 if seed is None else seed
        options = {name: getattr(self.agent, name, default) for name, default in (
            ("reward_mode", "individual"), ("team_reward_reduction", "mean"),
            ("team_return_mode", None), ("critic_mode", None))}
        return [RaceEnvFactory(
            self.scenario, self.scenario_dir, self.algorithm, self.run_id, worker,
            self.parallel_envs, self.reports.next_episode, (seed + worker * self.envs_per_worker) % 2**32,
            options, bool(self.reports.transition_hooks), self.focal_id,
            dict(measure_stages=self.measure_stages, measure_mpc=self.measure_mpc,
                 measure_hybrid=self.measure_hybrid, measure_cuda_sync=self.measure_cuda_sync),
            "cpu" if self.centralized_collection else str(self.collection_device), self.envs_per_worker,
            tuple(self._process_placement["cpu_affinity"]) if self._process_placement["cpu_affinity"] is not None else None)
            for worker in range(self.parallel_envs)]

    def _process_streams(self, batch):
        """Yield one chronological time stream per independent race slot."""
        expected_dims = 3 if self.envs_per_worker > 1 else 2
        if batch.ndim != expected_dims or batch.batch_size[0] != self.parallel_envs:
            raise RuntimeError(f"Unexpected process batch layout: {batch.batch_size}")
        if self.envs_per_worker > 1 and batch.batch_size[1] != self.envs_per_worker:
            raise RuntimeError(f"Unexpected grouped process batch layout: {batch.batch_size}")
        for worker in range(self.parallel_envs):
            for local in range(self.envs_per_worker):
                yield worker, local, (batch[worker, local] if self.envs_per_worker > 1
                                      else batch[worker])

    def _receive_process_batch(self, batch):
        """Aggregate actual worker decisions, including fixed-only continuations."""
        # The next-state tag is stamped at the action's environment step. The
        # root tag can still describe the previous batch's last observation.
        expected = self._updates - self._initial_updates
        if not (batch["next", "policy_version"] == expected).all():
            raise RuntimeError(f"Collected a stale policy batch; expected version {expected}")
        durations = []
        for worker, local, stream in self._process_streams(batch):
            if local == 0:
                packet = stream[0]["worker_profile"]
                self._receive_worker_profile(worker, packet)
                durations.append(packet["collection_seconds"])
            environment = worker * self.envs_per_worker + local
            if not (stream["environment_id"] == environment).all():
                raise RuntimeError(f"Unexpected environment ID in worker {worker}, slot {local}")
            episodes = stream["environment_episode"].squeeze(-1)
            starts = (episodes[1:] != episodes[:-1]).nonzero().flatten().add(1).tolist()
            if "race_events" in packet:
                events = dict(packet["race_events"].get(environment, ()))
            else:
                events = {index: value for index, value in enumerate(stream.get(("next", "race_events")).tolist())
                          if value}
            # Only reporting and episode boundaries require Python work. Prefix
            # sums preserve the exact counters visible to each hook, including
            # fixed-only decisions and rows without any reporting events.
            positions = sorted({0, len(stream) - 1, *starts, *events})
            counters = torch.cat((stream["next", "decision_physics_steps"],
                                  stream["next", "agent_steps"]), dim=-1).cumsum(dim=0)
            selected_counters = counters[positions].tolist()
            selected_episodes = episodes[positions].tolist()
            previous_index, previous_physics, previous_samples = -1, 0, 0
            for index, local_episode, (physics, samples) in zip(positions, selected_episodes, selected_counters):
                key = environment, local_episode
                self._worker_policy_starts.setdefault(key, self._updates)
                decisions = index - previous_index
                self._environment_steps += decisions
                self._physics_steps += physics - previous_physics
                self._agent_steps += samples - previous_samples
                self._pending_learning_steps += samples - previous_samples
                previous_index, previous_physics, previous_samples = index, physics, samples
                for enabled, counts in ((self.measure_mpc, self.mpc_environment_decisions),
                                        (self.measure_hybrid, self.hybrid_environment_decisions)):
                    if enabled:
                        counts[self._timing_phase] = counts.get(self._timing_phase, 0) + decisions
                for method, args in events.get(index, ()):
                    if method == "on_episode_start":
                        self.reports.next_episode = max(self.reports.next_episode, args[0]["episode"] + 1)
                    if method == "on_episode_end":
                        episode, reward, info, metrics = args
                        self.reports.completed += 1
                        race = metrics["race_record"]
                        race.update(environment_decisions_total=self._environment_steps,
                                    reported_at_environment_steps=self._environment_steps,
                                    policy_version_start=self._worker_policy_starts.pop(key),
                                    policy_version_end=self._updates,
                                    worker_id=worker)
                        metrics.update(self.counter_metrics())
                        self.reports.pending_episodes.append((episode, reward, info, metrics))
                    else:
                        hooks = self.reports.transition_hooks if method == "on_step" else self.hooks
                        for hook in hooks:
                            self.call_hook(hook, method, *args)
        self.worker_batch_metrics = {
            "perf/workers/batch_collection_min_seconds": min(durations),
            "perf/workers/batch_collection_max_seconds": max(durations),
            "perf/workers/batch_collection_mean_seconds": sum(durations) / len(durations),
            "perf/workers/batch_imbalance_ratio": max(durations) / max(sum(durations) / len(durations), 1e-9),
        }
        values = self.boundary_seconds[self._timing_phase]
        for name, value in (("worker_collection_sum", sum(durations)),
                            ("worker_collection_max", max(durations)),
                            ("worker_collection_min", min(durations)),
                            ("worker_wait", len(durations) * max(durations) - sum(durations))):
            values[name] = values.get(name, 0.) + value

    @staticmethod
    def _process_rollouts(pending):
        """Join time within environment slots and split physical races before GAE."""
        rollouts = []
        for chunks in pending.values():
            data = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
            active = data["agents", "active"].any(dim=(-2, -1))
            if not active.all():
                data = data[active]
            if not data.batch_size.numel():
                continue
            episodes = data["environment_episode"].squeeze(-1)
            boundaries = (episodes[1:] != episodes[:-1]).nonzero().flatten().add(1).tolist()
            start = 0
            for end in [*boundaries, len(data)]:
                rollouts.append(data[start:end])
                start = end
        return rollouts

    def train(self, n_episodes=0, *, total_steps=None):
        self._run_started, self._run_finished = time.perf_counter(), None
        self._process_placement = cpu_placement()
        self._process_placement.pop("cpu_topology")
        self._collection_batches = self._last_collection_batch_frames = 0
        self._steady_started = self._steady_finished = None
        self.boundary_seconds = {phase: {} for phase in self.stage_seconds}
        self.worker_stage_seconds = {phase: {} for phase in self.stage_seconds}
        self.worker_stage_calls = {phase: {} for phase in self.stage_seconds}
        self.worker_scope_seconds = {phase: {} for phase in self.stage_seconds}
        self.worker_metrics = {}
        self.worker_batch_metrics = {}
        self.inference_profile = {phase: {} for phase in self.stage_seconds}
        self.cuda_copy_seconds = 0.
        self.cuda_copy_calls = 0
        self.cuda_upload_seconds = 0.
        self.cuda_upload_calls = 0
        self._rate_baseline = self._rate_counters()
        self._initial_updates = self._updates
        self.process_cpu_seconds = dict.fromkeys(self.stage_seconds, 0.0)
        self._set_timing_phase("initialization")
        self.mpc_totals = {}
        self.mpc_environment_decisions = {}
        self.hybrid_totals = {}
        self.hybrid_environment_decisions = {}
        if total_steps is not None and (type(total_steps) is not int or total_steps <= 0):
            raise ValueError("total_steps must be a positive integer")
        if total_steps is None and (type(n_episodes) is not int or n_episodes < 0):
            raise ValueError("n_episodes must be a nonnegative integer")
        if self._resume_state is not None:
            from utils.torch_io import validate_checkpoint_compatibility
            validate_checkpoint_compatibility(self._resume_state["budget"], {
                "total_steps": total_steps, "n_episodes": n_episodes})
        for hook in self.hooks:
            bind = getattr(hook, "bind_trainer", None)
            if bind:
                bind(self)
        self.total_steps, self.n_episodes = total_steps, n_episodes
        self.env = NativeRaceTorchRLEnv(
            self.task, reward_mode=getattr(self.agent, "reward_mode", "individual"),
            team_reward_reduction=getattr(self.agent, "team_reward_reduction", "mean"))
        self.env.render_mode = "human" if self.render else None
        process = self.process_collection
        if not process:
            self.env.on_reset = self.reports.reset
            self.env.on_decision = self.reports.decision
            self.env.on_physics_step = self.reports.physics
        if self.measure_stages:
            self.initialization_seconds += time.perf_counter() - self._run_started
        pending = {} if process else []
        collection_started = time.perf_counter()
        collection_service_seconds = sum(self.service_seconds.values())
        metrics = {}
        collector = None
        central_env = None
        central_threads = None
        update_environment_steps = self._environment_steps
        update_physics_steps = self._physics_steps
        update_agent_steps = self._agent_steps
        self._set_progress()

        def report_update(updated, update_started):
            nonlocal collection_started, update_environment_steps, update_physics_steps, update_agent_steps
            nonlocal collection_service_seconds
            if updated:
                self._updates += 1
                self._optimizer_steps += int(updated["train/optimizer_steps"])
                updated.update({
                    **self.counter_metrics(),
                    **self.service_metrics(),
                    **self.performance_metrics(),
                    "train/update_environment_steps": self._environment_steps - update_environment_steps,
                    "train/update_physics_steps": self._physics_steps - update_physics_steps,
                    "train/update_agent_samples": self._agent_steps - update_agent_steps,
                    "perf/update_seconds": time.perf_counter() - update_started,
                    "perf/collection_seconds": max(0.0, update_started - collection_started - (
                        sum(self.service_seconds.values()) - collection_service_seconds)),
                })
                if process:
                    updated.update({
                        "train/update_sample_overshoot": max(0, updated["train/rollout_agent_samples"] - self.agent.n_steps),
                        "train/collection_batch_requested_frames": self.agent.frames_per_batch,
                        "train/collection_batch_frames": self._last_collection_batch_frames,
                        "train/collection_batch_rounding": max(0, self._last_collection_batch_frames - self.agent.frames_per_batch),
                    })
                update_environment_steps = self._environment_steps
                update_physics_steps = self._physics_steps
                update_agent_steps = self._agent_steps
                for hook in self.hooks:
                    self.call_hook(hook, "on_update", updated)
                collection_started = time.perf_counter()
                collection_service_seconds = sum(self.service_seconds.values())
            return updated

        def checkpoint_boundary():
            self.checkpoint_ready = True
            for hook in self.hooks:
                self.call_hook(hook, "on_checkpoint_boundary")

        def optimize(data):
            self._set_progress()
            def sync_update():
                if self.measure_stages and torch.device(self.agent.device).type == "cuda":
                    self.measure_stage("cuda_synchronization", synchronize_device, self.agent.device)
                else:
                    synchronize_device(self.agent.device)
            sync_update()
            update_started = time.perf_counter()
            update = self.agent.update_rollouts if process else self.agent.update
            updated = (self.measure_stage("update", update, data, device=self.agent.device)
                       if self.measure_stages else update(data))
            # The trainer owns cadence, including final rollouts smaller than
            # the legacy PPO minimum used by direct update_rollouts callers.
            if not updated and getattr(self.agent, "_pending_batches", []):
                updated = (self.measure_stage("update", self.agent.flush_pending_update, device=self.agent.device)
                           if self.measure_stages else self.agent.flush_pending_update())
            sync_update()
            self._record_boundary("update", update_started)
            if updated and (process or self.collection_device != self.agent.device):
                # Synchronous workers are idle here. Publish the new actor before
                # any next() call; native version tags verify worker application.
                sync_started = time.perf_counter()
                # The TensorDict form works for both direct policy copies and
                # process collectors in TorchRL 0.14 (the direct module form
                # is not converted to weights by its fallback updater).
                collector.update_policy_weights_(TensorDict.from_module(self.agent.collection_policy).data)
                self._record_boundary("weight_sync", sync_started)
            return report_update(updated, update_started)

        stage_resources = ExitStack()
        try:
            if self.measure_mpc and not process:
                self._instrument_mpc(stage_resources)
            if self.measure_hybrid and not process:
                self._instrument_hybrid(stage_resources)
            self._instrument_stages(stage_resources)
            require_source = getattr(self.agent, "_require_lora_source", None)
            if require_source:
                require_source()
            if not self.budget_reached():
                # Serial interrupts at an exact decision boundary. Processes
                # consume whole synchronous batches and report their overshoot.
                # Neither path introduces artificial trajectory truncations.
                if self._resume_state is not None:
                    self.env.set_seed(self._resume_state["recovery_seed"])
                def create_collector():
                    nonlocal central_env, central_threads
                    if self.centralized_collection:
                        from training.process_collection import reset_inference_profile, inference_profile
                        central_threads = torch.get_num_threads()
                        # Shared buffers avoid per-step tensor serialization.
                        # Metadata probes reuse factories that seed their process;
                        # preserve the parent's policy RNG while they run.
                        with preserve_random_state():
                            central_env = ParallelEnv(
                                self.parallel_envs, [partial(factory) for factory in self._process_factories()],
                                use_buffers=True, mp_start_method="spawn", serial_for_single=False,
                                num_threads=central_threads,
                                num_sub_threads=self.scenario["experiment"].get("torch_threads", 1))

                        def receive_reports(batch):
                            if self.measure_stages:
                                self._merge_profile(self.inference_profile[self._timing_phase], inference_profile())
                            packets = central_env.end_collection()
                            batch["worker_profile"] = torch.stack([
                                NonTensorData(packet, batch_size=batch.batch_size[1:])
                                for packet in packets], dim=0)

                        def begin_batch():
                            reset_inference_profile()
                            central_env.begin_collection()

                        return Collector(
                            TransformedEnv(central_env), self.agent.collection_policy,
                            backend="direct", frames_per_batch=self.agent.frames_per_batch,
                            total_frames=-1, policy_device=self.collection_device,
                            env_device="cpu", storing_device="cpu",
                            no_cuda_sync=not self.measure_cuda_sync,
                            reset_at_each_iter=False, set_truncated=False, use_buffers=False,
                            auto_register_policy_transforms=False, track_policy_version=True,
                            pre_collect_hook=begin_batch, post_collect_hook=receive_reports)
                    if process:
                        from training.process_collection import (
                            pre_collect_profile, post_collect_profile, instrument_policy_inference)
                        if self.measure_stages:
                            instrument_policy_inference(self.agent.collection_policy, stage_resources,
                                                        cuda_sync=self.measure_cuda_sync)
                        return Collector(
                            self._process_factories(), self.agent.collection_policy,
                            backend="process", num_collectors=self.parallel_envs, sync=True,
                            frames_per_batch=self.agent.frames_per_batch, total_frames=-1,
                            policy_device=self.collection_device, env_device="cpu", storing_device="cpu",
                            no_cuda_sync=not self.measure_cuda_sync,
                            reset_at_each_iter=False, set_truncated=False, use_buffers=False,
                            auto_register_policy_transforms=False, track_policy_version=True,
                            pre_collect_hook=pre_collect_profile, post_collect_hook=post_collect_profile,
                            backend_options={"cat_results": "stack", "num_sub_threads": self.scenario["experiment"].get("torch_threads", 1)})
                    return Collector(
                        self.env, self.agent.collection_policy, backend="direct",
                        frames_per_batch=self.agent.frames_per_batch, total_frames=-1,
                        policy_device=self.collection_device, env_device="cpu", storing_device="cpu",
                        no_cuda_sync=not self.measure_cuda_sync,
                        reset_at_each_iter=False, set_truncated=False, use_buffers=True,
                        auto_register_policy_transforms=False,
                        # Buffered interruption pads unused rows with traj_ids=-1.
                        postproc=lambda batch: batch[batch["collector", "traj_ids"] >= 0],
                        interruptor=_CollectionBoundary(self))
                startup_started = time.perf_counter()
                collector = (self.measure_stage("collector_initialize", create_collector)
                             if self.measure_stages else create_collector())
                if self.centralized_collection and self.measure_stages:
                    from training.process_collection import instrument_policy_inference
                    instrument_policy_inference(collector.policy, stage_resources,
                                                cuda_sync=self.measure_cuda_sync)
                if not process and self.measure_stages and self.collection_device != self.agent.device:
                    # Instrument the copy only after TorchRL creates it; wrapping
                    # the source's forward would survive deepcopy with a bound
                    # reference to the learner on the wrong device.
                    original_forward = collector.policy.forward
                    collector.policy.forward = lambda *args, **kwargs: self.measure_stage(
                        "policy_inference", original_forward, *args, device=self.collection_device, **kwargs)
                    stage_resources.callback(setattr, collector.policy, "forward", original_forward)
                self._record_boundary("worker_startup" if process else "collector_startup", startup_started)
                self._set_timing_phase("warmup" if self.profiling_warmup_updates else "steady")
                if self._timing_phase == "steady":
                    self._steady_started, self._steady_baseline = time.perf_counter(), self._rate_counters()
                if self._resume_state is not None:
                    from agents.common.checkpoints import restore_rng
                    restore_rng(self._resume_state["rng"])
                iterator = iter(collector)
                while True:
                    try:
                        batch_started = time.perf_counter()
                        with CudaCollectionProfile((not process or self.centralized_collection)
                                                   and self.measure_stages and self.measure_cuda_sync
                                                   and self.collection_device.type == "cuda") as copy_profile:
                            data = (self.measure_stage("collector", next, iterator)
                                    if self.measure_stages else next(iterator))
                        if copy_profile.enabled:
                            self.cuda_copy_seconds += copy_profile.device_to_cpu_seconds
                            self.cuda_copy_calls += copy_profile.device_to_cpu_calls
                            self.cuda_upload_seconds += copy_profile.cpu_to_device_seconds
                            self.cuda_upload_calls += copy_profile.cpu_to_device_calls
                        self._record_boundary("collection", batch_started)
                    except StopIteration:
                        break
                    self.checkpoint_ready = False
                    self._collection_batches += 1
                    self._last_collection_batch_frames = data.numel()
                    updated = False
                    metrics = {}
                    aggregation_started = time.perf_counter()
                    # Fixed-only continuation counts toward the physical budget,
                    # but follows the learner endpoint and needs no GAE rows.
                    if process:
                        self._receive_process_batch(data)
                        # Own the retained tensors before the next collector
                        # batch; per-environment slices can then safely be views.
                        owned = data.exclude("race_events", ("next", "race_events"), "worker_profile").clone()
                        for worker, local, fragment in self._process_streams(owned):
                            environment = worker * self.envs_per_worker + local
                            pending.setdefault(environment, []).append(fragment)
                    else:
                        data = data[data["agents", "active"].any(dim=(-2, -1))]
                        if data.batch_size.numel():
                            pending.append(data)
                    self._record_boundary("aggregation", aggregation_started)
                    ready = self._pending_learning_steps >= self.agent.n_steps or self.budget_reached()
                    if ready and pending:
                        prepare_started = time.perf_counter()
                        rollouts = self._process_rollouts(pending) if process else torch.cat(pending, dim=0)
                        self._record_boundary("aggregation", prepare_started)
                        metrics = optimize(rollouts)
                        pending, self._pending_learning_steps = ({} if process else []), 0
                        updated = bool(metrics)
                    self.reports.dispatch(metrics)
                    if updated:
                        checkpoint_boundary()
                        if (self._timing_phase == "warmup"
                                and self._updates - self._initial_updates >= self.profiling_warmup_updates):
                            self._set_timing_phase("steady")
                            self._steady_started, self._steady_baseline = time.perf_counter(), self._rate_counters()
                    if self.budget_reached() or self._should_stop():
                        break
                if not self._should_stop() and pending:
                    prepare_started = time.perf_counter()
                    rollouts = self._process_rollouts(pending) if process else torch.cat(pending, dim=0)
                    self._record_boundary("aggregation", prepare_started)
                    metrics = optimize(rollouts)
                    pending, self._pending_learning_steps = ({} if process else []), 0
                    if metrics:
                        checkpoint_boundary()
            self._steady_finished = time.perf_counter()
            for hook in self.hooks:
                self.call_hook(hook, "on_training_end")
        finally:
            stage_resources.close()
            shutdown_started = time.perf_counter()
            try:
                if collector is not None:
                    collector.shutdown()
            finally:
                try:
                    if central_env is not None and not central_env.is_closed:
                        central_env.close()
                finally:
                    try:
                        if process or collector is None:
                            self.env.close()
                    finally:
                        # ParallelEnv.close adjusts the parent's thread pool.
                        if central_threads is not None:
                            torch.set_num_threads(central_threads)
            self._record_boundary("shutdown", shutdown_started)
            self._run_finished = time.perf_counter()
            self._finish_process_phase()
            if self.measure_stages:
                self._sample_peak_rss()
