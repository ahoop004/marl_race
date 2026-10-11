"""Direct TorchRL collection and a small shared PPO/MAPPO update loop."""
import time
import math
from contextlib import ExitStack
from functools import wraps

import torch
from torchrl.collectors import Collector

from adapters.native_torchrl import NativeRaceTorchRLEnv
from training.contracts import validate_team_reward_composers
from training.race_reporting import RaceTrainingReports
from utils.torch_io import synchronize_device


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
                 profiling_warmup_updates=1):
        self.task, self.agent = task, agent
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
        self.profiling_warmup_updates = profiling_warmup_updates
        self.stage_seconds = {phase: {} for phase in ("initialization", "warmup", "steady")}
        self.stage_calls = {phase: {} for phase in self.stage_seconds}
        self._timing_phase = "initialization"
        self.initialization_seconds = 0.0
        self._run_started = self._run_finished = self._steady_started = None
        self._rate_baseline = self._steady_baseline = None
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
        # n_steps targets active actor samples. A joint decision is indivisible,
        # so MAPPO can exceed the target by at most number_of_learners - 1.
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
        """Exclusive wall time; detailed CUDA timing synchronizes at stage boundaries."""
        if not self.measure_stages:
            return callback(*args, **kwargs)
        phase = self._timing_phase
        if device is not None:
            synchronize_device(device)
        previous = self._stage_total() + sum(self.service_seconds.values())
        started = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            if device is not None:
                synchronize_device(device)
            nested = self._stage_total() + sum(self.service_seconds.values()) - previous
            values, calls = self.stage_seconds[phase], self.stage_calls[phase]
            values[category] = values.get(category, 0.0) + max(0.0, time.perf_counter() - started - nested)
            calls[category] = calls.get(category, 0) + 1

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
                return self.measure_stage(category, original, *args, device=device, **kwargs)
            setattr(owner, name, timed)
            def restore():
                if had_value:
                    setattr(owner, name, own_value)
                else:
                    delattr(owner, name)
            resources.callback(restore)

        instrument(self.env, "step", "environment_step")
        instrument(self.env, "reset", "environment_reset")
        instrument(self.env, "_output", "tensor_transfer")
        instrument(self.agent.collection_policy, "forward", "policy_inference", self.agent.device)
        instrument(self.agent.gae, "forward", "gae", self.agent.device)
        for composer in (*self.task.obs_composers.values(), *self.task.reward_composers.values()):
            instrument(composer, "wrap" if hasattr(composer, "wrap") else "compute", "observation_reward")
        for controller in self.task.fixed_controllers.values():
            instrument(controller, "act", "fixed_controller")
        for name in ("_split_obs", "_update_centerline_observation_facts", "_attach_central_state"):
            instrument(self.task.env, name, "observation_reward")
        simulator = getattr(self.task.env, "sim", None)
        if simulator is not None:
            instrument(simulator, "step", "physics")
            instrument(simulator, "refresh_scans", "lidar")
            instrument(simulator.scan_simulator, "scan", "lidar")
            for vehicle in simulator.agents:
                instrument(vehicle, "ray_cast_agents", "lidar")
        self.agent.profile_call = self.measure_stage
        self.env.profile_call = self.measure_stage
        resources.callback(setattr, self.agent, "profile_call", None)
        resources.callback(setattr, self.env, "profile_call", None)

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
        if self.measure_stages:
            result.update({f"perf/stages/{phase}/{name}_seconds": seconds
                for phase, values in self.stage_seconds.items() for name, seconds in values.items()})
            result.update({f"perf/stages/{phase}/{name}_calls": calls
                for phase, values in self.stage_calls.items() for name, calls in values.items()})
            result["perf/initialization_seconds"] = self.initialization_seconds + sum(self.stage_seconds["initialization"].values())
            result["perf/warmup_seconds"] = sum(self.stage_seconds["warmup"].values())
            if self._steady_started is not None:
                duration = max(now - self._steady_started, 1e-9)
                result.update({f"perf/steady_{name}_per_second": (value - base) / duration
                    for name, value, base in zip(names, self._rate_counters(), self._steady_baseline)})
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
        return {"train/environment_steps": self._environment_steps,
                "train/physics_steps": self._physics_steps,
                "train/agent_steps": self._agent_steps,
                "train/completed_episodes": self.reports.completed,
                "train/updates": self._updates,
                "train/optimizer_steps_total": self._optimizer_steps}

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

    def train(self, n_episodes=0, *, total_steps=None):
        self._run_started, self._run_finished = time.perf_counter(), None
        self._rate_baseline = self._rate_counters()
        self._initial_updates = self._updates
        self._timing_phase = "initialization"
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
        self.env.on_reset = self.reports.reset
        self.env.on_decision = self.reports.decision
        self.env.on_physics_step = self.reports.physics
        if self.measure_stages:
            self.initialization_seconds += time.perf_counter() - self._run_started
        pending = []
        collection_started = time.perf_counter()
        collection_service_seconds = sum(self.service_seconds.values())
        metrics = {}
        collector = None
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
            synchronize_device(self.agent.device)
            update_started = time.perf_counter()
            updated = (self.measure_stage("update", self.agent.update, data, device=self.agent.device)
                       if self.measure_stages else self.agent.update(data))
            # The trainer owns cadence, including final rollouts smaller than
            # the legacy PPO minimum used by direct update_rollouts callers.
            if not updated and getattr(self.agent, "_pending_batches", []):
                updated = (self.measure_stage("update", self.agent.flush_pending_update, device=self.agent.device)
                           if self.measure_stages else self.agent.flush_pending_update())
            synchronize_device(self.agent.device)
            return report_update(updated, update_started)

        stage_resources = ExitStack()
        try:
            if self.measure_mpc:
                self._instrument_mpc(stage_resources)
            if self.measure_hybrid:
                self._instrument_hybrid(stage_resources)
            if self.measure_stages:
                self._instrument_stages(stage_resources)
            require_source = getattr(self.agent, "_require_lora_source", None)
            if require_source:
                require_source()
            if not self.budget_reached():
                # An unbounded collector plus its interruption protocol avoids
                # 0.14's rounding of non-divisible total_frames. No private
                # collector state, manual stepping, or artificial truncation.
                if self._resume_state is not None:
                    self.env.set_seed(self._resume_state["recovery_seed"])
                def create_collector():
                    return Collector(
                        self.env, self.agent.collection_policy, backend="direct",
                        frames_per_batch=self.agent.frames_per_batch, total_frames=-1,
                        policy_device=self.agent.device, env_device="cpu", storing_device="cpu",
                        reset_at_each_iter=False, set_truncated=False, use_buffers=True,
                        auto_register_policy_transforms=False,
                        # Buffered interruption pads unused rows with traj_ids=-1.
                        postproc=lambda batch: batch[batch["collector", "traj_ids"] >= 0],
                        interruptor=_CollectionBoundary(self))
                collector = (self.measure_stage("collector_initialize", create_collector)
                             if self.measure_stages else create_collector())
                self._timing_phase = "warmup" if self.profiling_warmup_updates else "steady"
                if self.measure_stages and self._timing_phase == "steady":
                    self._steady_started, self._steady_baseline = time.perf_counter(), self._rate_counters()
                if self._resume_state is not None:
                    from agents.common.checkpoints import restore_rng
                    restore_rng(self._resume_state["rng"])
                iterator = iter(collector)
                while True:
                    try:
                        data = (self.measure_stage("collector", next, iterator)
                                if self.measure_stages else next(iterator))
                    except StopIteration:
                        break
                    self.checkpoint_ready = False
                    updated = False
                    metrics = {}
                    # Fixed-only continuation counts toward the physical budget,
                    # but follows the learner endpoint and needs no GAE rows.
                    data = data[data["agents", "active"].any(dim=(-2, -1))]
                    if data.batch_size.numel():
                        pending.append(data)
                    ready = self._pending_learning_steps >= self.agent.n_steps or self.budget_reached()
                    if ready and pending:
                        metrics = optimize(torch.cat(pending, dim=0))
                        pending, self._pending_learning_steps = [], 0
                        updated = bool(metrics)
                    self.reports.dispatch(metrics)
                    if updated:
                        checkpoint_boundary()
                        if ((self.measure_stages or self.measure_mpc or self.measure_hybrid) and self._timing_phase == "warmup"
                                and self._updates - self._initial_updates >= self.profiling_warmup_updates):
                            self._timing_phase = "steady"
                            self._steady_started, self._steady_baseline = time.perf_counter(), self._rate_counters()
                    if self.budget_reached() or self._should_stop():
                        break
                if not self._should_stop() and pending:
                    metrics = optimize(torch.cat(pending, dim=0))
                    pending, self._pending_learning_steps = [], 0
                    if metrics:
                        checkpoint_boundary()
            for hook in self.hooks:
                self.call_hook(hook, "on_training_end")
        finally:
            stage_resources.close()
            if collector is not None:
                collector.shutdown()
            else:
                self.env.close()
            self._run_finished = time.perf_counter()
