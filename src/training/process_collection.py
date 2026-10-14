"""Serializable race construction and reports carried by process collector batches."""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from contextlib import ExitStack
import time
import sys

import torch
from tensordict import NonTensorData
from torchrl.envs import SerialEnv, TransformedEnv
from torchrl.collectors import Collector
from torchrl.weight_update import SharedMemWeightSyncScheme
from torchrl import logger as _logger

try:
    import resource
except ImportError:
    resource = None

from adapters.native_torchrl import NativeRaceTorchRLEnv
from training.race_reporting import RaceTrainingReports
from training.runtime import seed_process, CudaCollectionProfile
from utils.cpu_affinity import cpu_placement
from utils.torch_io import synchronize_device


# Collector hooks execute in the same worker that calls the environment factory.
# This reference is process-local; reports use TorchRL's existing batch transport.
_worker_reports = {}
_worker_started = _collection_started = _collection_cpu_started = None
_collection_wall = _collection_cpu = 0.
_factory_seconds = 0.
_cuda_profile = None
_worker_placement = {}
_inference_profile = {}
_inference_started = None
_worker_startup = {}


def begin_worker_startup():
    """Supported worker init_fn; excludes interpreter imports before this hook."""
    global _worker_startup
    _worker_startup = {"process_started": time.perf_counter()}


class StartupProfileCollector(Collector):
    """Time the runner's policy factory without changing its weight handshake."""
    def __init__(self, *args, **kwargs):
        factory = kwargs["policy_factory"]
        def policy_factory():
            started = time.perf_counter()
            _worker_startup["initial_weight_receive_started_seconds"] = (
                started - _worker_startup["process_started"])
            policy = factory()
            _worker_startup["initial_weight_receive_seconds"] = time.perf_counter() - started
            _worker_startup["initial_weights_received_seconds"] = time.perf_counter() - _worker_startup["process_started"]
            return policy
        kwargs["policy_factory"] = policy_factory
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            _logger.exception("Collector worker %s initialization failed after %.3fs (stages=%s)",
                              kwargs.get("worker_idx"),
                              time.perf_counter() - _worker_startup["process_started"],
                              {k: v for k, v in _worker_startup.items() if k != "process_started"})
            raise
        _worker_startup["policy_ready_seconds"] = time.perf_counter() - _worker_startup["process_started"]


class StartupProfileWeightSync(SharedMemWeightSyncScheme):
    """Only time supported sender hooks; keep the default shared transport."""
    def __init__(self):
        super().__init__()
        self.startup = {}

    def init_on_sender(self, **kwargs):
        super().init_on_sender(**kwargs)
        self.startup["spawning_started"] = time.perf_counter()
        _logger.info("Collector worker spawning begins")

    def connect(self, **kwargs):
        if not self.initialized_on_sender:
            return super().connect(**kwargs)
        started = time.perf_counter()
        self.startup["worker_spawning_seconds"] = started - self.startup["spawning_started"]
        self.startup["spawned_workers"] = len(self.context.procs)
        _logger.info("Collector spawned %s workers in %.3fs; distributing initial weights",
                     self.startup["spawned_workers"], self.startup["worker_spawning_seconds"])
        super().connect(**kwargs)
        # Queue puts return before workers receive/apply weights. Readiness is
        # established later by the unmodified MultiSyncCollector constructor.
        self.startup["initial_weight_dispatch_seconds"] = time.perf_counter() - started
        self.startup["weight_dispatch_finished"] = time.perf_counter()
        _logger.info("Initial weight dispatch finished in %.3fs; waiting for worker readiness",
                     self.startup["initial_weight_dispatch_seconds"])


def reset_inference_profile():
    global _inference_profile
    _inference_profile = dict(calls=0, environment_rows=0, agent_rows=0, seconds=0.)


def inference_profile():
    return dict(_inference_profile)


def begin_policy_inference(module, args, *, cuda_sync=False):
    global _inference_started
    if not _inference_profile:
        reset_inference_profile()
    data = args[0]
    if cuda_sync:
        synchronize_device(data["agents", "observation"].device)
    _inference_started = time.perf_counter()
    _inference_profile["calls"] += 1
    _inference_profile["environment_rows"] += data.numel()
    _inference_profile["agent_rows"] += data["agents"].numel()


def end_policy_inference(module, args, result, *, cuda_sync=False):
    if cuda_sync:
        synchronize_device(args[0]["agents", "observation"].device)
    _inference_profile["seconds"] += time.perf_counter() - _inference_started


def instrument_policy_inference(policy, resources, *, cuda_sync=False):
    """Serializable hooks measure the actual collector policy, including copies."""
    reset_inference_profile()
    resources.callback(policy.register_forward_pre_hook(
        partial(begin_policy_inference, cuda_sync=cuda_sync)).remove)
    resources.callback(policy.register_forward_hook(
        partial(end_policy_inference, cuda_sync=cuda_sync)).remove)


def pre_collect_profile():
    global _collection_started, _collection_cpu_started, _cuda_profile
    _collection_started, _collection_cpu_started = time.perf_counter(), time.process_time()
    reset_inference_profile()
    for reports in _worker_reports.values():
        reports.batch_events = []
        reports.batch_start = reports.trainer._environment_steps
        if reports.profiler is not None:
            reports.profiler._timing_phase = "steady"
    _cuda_profile = CudaCollectionProfile(any(
        reports.profile_cuda_collection for reports in _worker_reports.values()))
    _cuda_profile.__enter__()


def worker_collection_report():
    global _collection_wall, _collection_cpu
    wall = time.perf_counter() - _collection_started
    cpu = time.process_time() - _collection_cpu_started
    _cuda_profile.__exit__(None, None, None)
    first = next(iter(_worker_reports.values()))
    first.cuda_copy_seconds += _cuda_profile.device_to_cpu_seconds
    first.cuda_copy_calls += _cuda_profile.device_to_cpu_calls
    first.cuda_upload_seconds += _cuda_profile.cpu_to_device_seconds
    first.cuda_upload_calls += _cuda_profile.cpu_to_device_calls
    _collection_wall += wall
    _collection_cpu += cpu
    process = dict(collection_seconds=_collection_wall, collection_cpu_seconds=_collection_cpu,
                   collection_cpu_percent=100 * _collection_cpu / max(_collection_wall, 1e-9),
                   cpu_seconds=time.process_time(), factory_seconds=_factory_seconds,
                   environment_steps=sum(r.trainer._environment_steps for r in _worker_reports.values()),
                   learner_samples=sum(r.trainer._agent_steps for r in _worker_reports.values()),
                   completed_episodes=sum(r.reports.completed for r in _worker_reports.values()))
    process.update(_worker_placement)
    process.update({"startup/" + key: value for key, value in _worker_startup.items()
                    if key != "process_started"})
    from agents.common.compilation import compilation_metrics
    process.update({"compilation/" + key: value for key, value in compilation_metrics().items()})
    if _cuda_profile.enabled:
        process.update(collection_device_to_cpu_seconds=first.cuda_copy_seconds,
                       collection_device_to_cpu_calls=first.cuda_copy_calls,
                       collection_cpu_to_device_seconds=first.cuda_upload_seconds,
                       collection_cpu_to_device_calls=first.cuda_upload_calls)
    if resource is not None:
        process["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    # Sparse chronological reports share the existing batch transport. Empty
    # transitions need no NonTensorData allocation, stacking, or serialization.
    packet = dict(collection_seconds=wall, process=process,
                  inference=inference_profile(),
                  race_events={environment: reports.batch_events
                               for environment, reports in _worker_reports.items()
                               if reports.batch_events})
    for reports in _worker_reports.values():
        profiler = reports.profiler
        if profiler is not None:
            for name in ("stage_seconds", "stage_calls", "scope_seconds", "mpc_totals", "hybrid_totals"):
                profiler._merge_profile(packet.setdefault(name.removesuffix("_totals"), {}), getattr(profiler, name))
                setattr(profiler, name, {phase: {} for phase in profiler.stage_seconds}
                        if not name.endswith("_totals") else {})
    return packet


def post_collect_profile(batch):
    batch["worker_profile"] = NonTensorData(worker_collection_report(), batch_size=batch.batch_size)


class _WorkerReports:
    """Reuse race reporting locally; send hook arguments, never execute user hooks."""
    def __init__(self, env, *, algorithm, run_id, worker, workers, episode_offset,
                 agent_options, transitions, focal_id, profiling=None, collection_device="cpu"):
        self.env = env
        self.batch_events = []
        self.batch_start = 0
        self.requires_transition_record = transitions
        self.trainer = SimpleNamespace(
            task=env.task, agent=SimpleNamespace(**agent_options), env=env,
            algorithm=algorithm, run_id=run_id, focal_id=focal_id,
            hooks=[self], environment_id=worker, episode_stride=workers,
            _updates=0, _environment_steps=0, _physics_steps=0, _agent_steps=0,
            _pending_learning_steps=0, measure_mpc=False, measure_hybrid=False,
            budget_reached=lambda: False, _should_stop=lambda: False,
            call_hook=lambda hook, method, *args: getattr(hook, method)(*args),
            counter_metrics=lambda: {})
        self.profiler = None
        self.collection_device = collection_device
        self.cuda_copy_seconds = 0.
        self.cuda_copy_calls = 0
        self.cuda_upload_seconds = 0.
        self.cuda_upload_calls = 0
        self.profile_cuda_collection = (collection_device.startswith("cuda") and bool(profiling)
                                       and profiling.get("measure_stages", False)
                                       and profiling.get("measure_cuda_sync", False))
        if profiling and any(profiling.values()):
            # Reuse only the trainer's instrumentation; no worker learner or
            # optimization loop is created.
            from training.on_policy import OnPolicyTrainer
            agent = SimpleNamespace(**agent_options, agent_ids=list(env.agent_ids),
                                    collection_policy=None, gae=None, device=torch.device("cpu"))
            self.trainer = self.profiler = OnPolicyTrainer(
                env.task, agent, hooks=[self], algorithm=algorithm, run_id=run_id,
                focal_agent_id=focal_id, collection_device=collection_device, **profiling)
            self.trainer.env = env
            env.profile_resources = ExitStack()
            if profiling["measure_stages"]:
                self.profiler._instrument_stages(env.profile_resources)
            if profiling["measure_mpc"]:
                self.profiler._instrument_mpc(env.profile_resources)
            if profiling["measure_hybrid"]:
                self.profiler._instrument_hybrid(env.profile_resources)
        self.reports = self.profiler.reports if self.profiler else RaceTrainingReports(self.trainer)
        self.reports.environment_id, self.reports.episode_stride = worker, workers
        self.reports.next_episode = episode_offset + worker

    def on_episode_start(self, metadata):
        metadata.update(environment_id=self.env.environment_id,
                        environment_episode=self.env.environment_episode,
                        episode=self.reports.episode, worker_id=self.worker_id)
        self.env.report_events.append(("on_episode_start", (metadata,)))

    def on_step(self, record):
        record.info.update(environment_id=self.env.environment_id,
                           environment_episode=self.env.environment_episode,
                           worker_id=self.worker_id)
        self.env.report_events.append(("on_step", (record,)))

    def on_episode_end(self, *args):
        self.env.report_events.append(("on_episode_end", args))

    def decision(self, result):
        self.reports.decision(result)
        self.reports.dispatch({})
        if self.env.batch_reports and self.env.report_events:
            index = self.trainer._environment_steps - self.batch_start - 1
            self.batch_events.append((index, self.env.report_events))
            self.env.report_events = []


@dataclass
class RaceEnvFactory:
    """Only configuration crosses the boundary; each process owns its race slots."""
    scenario: dict
    scenario_dir: str | None
    algorithm: str
    run_id: str
    worker: int
    workers: int
    episode_offset: int
    seed: int
    agent_options: dict
    transitions: bool
    focal_id: str
    profiling: dict | None = None
    collection_device: str = "cpu"
    envs_per_worker: int = 1
    parent_cpu_affinity: tuple | None = None

    def __call__(self):
        global _worker_reports, _worker_started, _collection_wall, _collection_cpu, _factory_seconds
        global _worker_placement
        _worker_started = time.perf_counter()
        _collection_wall = _collection_cpu = 0.
        _worker_reports = {}
        _worker_placement = cpu_placement()
        # Topology is inspected once at construction, not on every decision.
        _worker_placement.pop("cpu_topology")
        if self.parent_cpu_affinity is not None and _worker_placement["cpu_affinity"] is not None:
            _worker_placement["cpu_affinity_within_parent"] = set(
                _worker_placement["cpu_affinity"]).issubset(self.parent_cpu_affinity)
            _worker_placement["cpu_affinity_matches_parent"] = set(
                _worker_placement["cpu_affinity"]) == set(self.parent_cpu_affinity)
            if not _worker_placement["cpu_affinity_within_parent"]:
                raise RuntimeError("Collector CPU affinity exceeds the parent allocation")
        torch.set_num_threads(self.scenario["experiment"].get("torch_threads", 1))
        seed_process(self.seed)
        if self.envs_per_worker == 1:
            env = self._make_env(self.worker, self.seed)
        else:
            env = SerialEnv(self.envs_per_worker, [
                partial(self._make_env, self.worker * self.envs_per_worker + local,
                        (self.seed + local) % 2**32)
                for local in range(self.envs_per_worker)], use_buffers=False)
            try:
                # SerialEnv first probes temporary environments for specs.
                # Starting it replaces those reporters with the live instances.
                env.start()
            except BaseException:
                env.close(raise_if_closed=False)
                raise
            # TorchRL 0.14 rejects automatic policy-version tracking on a
            # bare BatchedEnvBase. An outer TransformedEnv lets the collector
            # own and advance its usual PolicyVersion transform for all slots.
            env = TransformedEnv(env)
        _factory_seconds = time.perf_counter() - _worker_started
        if _worker_startup:
            _worker_startup["environment_construction_seconds"] = _factory_seconds
            _worker_startup["environment_ready_seconds"] = time.perf_counter() - _worker_startup["process_started"]
        # ParallelEnv dispatches these public methods to each worker once per
        # collection batch, reusing the process collector's sparse transport.
        env.begin_collection = pre_collect_profile
        env.end_collection = worker_collection_report
        return env

    def _make_env(self, environment_id, seed):
        scenario = deepcopy(self.scenario)
        scenario["experiment"]["seed"] = seed
        env = NativeRaceTorchRLEnv.from_scenario(
            scenario, scenario_dir=self.scenario_dir, environment_id=environment_id,
            reward_mode=self.agent_options["reward_mode"],
            team_reward_reduction=self.agent_options["team_reward_reduction"], batch_reports=True)
        try:
            reports = _WorkerReports(
                env, algorithm=self.algorithm, run_id=self.run_id, worker=environment_id,
                workers=self.workers * self.envs_per_worker, episode_offset=self.episode_offset,
                agent_options=self.agent_options, transitions=self.transitions,
                focal_id=self.focal_id, profiling=self.profiling, collection_device=self.collection_device)
            reports.worker_id = self.worker
            _worker_reports[environment_id] = reports
            env.on_reset = reports.reports.reset
            env.on_physics_step = reports.reports.physics
            env.on_decision = reports.decision
            return env
        except BaseException:
            env.close()
            raise
