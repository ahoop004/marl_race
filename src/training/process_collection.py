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

try:
    import resource
except ImportError:
    resource = None

from adapters.native_torchrl import NativeRaceTorchRLEnv
from training.race_reporting import RaceTrainingReports
from training.runtime import seed_process, CudaCollectionProfile


# Collector hooks execute in the same worker that calls the environment factory.
# This reference is process-local; reports use TorchRL's existing batch transport.
_worker_reports = {}
_worker_started = _collection_started = _collection_cpu_started = None
_collection_wall = _collection_cpu = 0.
_factory_seconds = 0.
_cuda_profile = None


def pre_collect_profile():
    global _collection_started, _collection_cpu_started, _cuda_profile
    _collection_started, _collection_cpu_started = time.perf_counter(), time.process_time()
    for reports in _worker_reports.values():
        reports.batch_events = []
        reports.batch_start = reports.trainer._environment_steps
        if reports.profiler is not None:
            reports.profiler._timing_phase = "steady"
    _cuda_profile = CudaCollectionProfile(any(
        reports.profile_cuda_collection for reports in _worker_reports.values()))
    _cuda_profile.__enter__()


def post_collect_profile(batch):
    global _collection_wall, _collection_cpu
    wall = time.perf_counter() - _collection_started
    cpu = time.process_time() - _collection_cpu_started
    _cuda_profile.__exit__(None, None, None)
    first = next(iter(_worker_reports.values()))
    first.cuda_copy_seconds += _cuda_profile.device_to_cpu_seconds
    first.cuda_copy_calls += _cuda_profile.device_to_cpu_calls
    _collection_wall += wall
    _collection_cpu += cpu
    process = dict(collection_seconds=_collection_wall, collection_cpu_seconds=_collection_cpu,
                   collection_cpu_percent=100 * _collection_cpu / max(_collection_wall, 1e-9),
                   cpu_seconds=time.process_time(), factory_seconds=_factory_seconds,
                   environment_steps=sum(r.trainer._environment_steps for r in _worker_reports.values()),
                   learner_samples=sum(r.trainer._agent_steps for r in _worker_reports.values()),
                   completed_episodes=sum(r.reports.completed for r in _worker_reports.values()))
    if _cuda_profile.enabled:
        process.update(collection_device_to_cpu_seconds=first.cuda_copy_seconds,
                       collection_device_to_cpu_calls=first.cuda_copy_calls)
    if resource is not None:
        process["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    # Sparse chronological reports share the existing batch transport. Empty
    # transitions need no NonTensorData allocation, stacking, or serialization.
    packet = dict(collection_seconds=wall, process=process,
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
    batch["worker_profile"] = NonTensorData(packet, batch_size=batch.batch_size)


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

    def __call__(self):
        global _worker_reports, _worker_started, _collection_wall, _collection_cpu, _factory_seconds
        _worker_started = time.perf_counter()
        _collection_wall = _collection_cpu = 0.
        _worker_reports = {}
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
