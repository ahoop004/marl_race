"""Opt-in serial measurements using the collection profiler's callback protocol.

Instance wrappers are installed only for a profiling session and restored even
on failure. Kernels receive one shared, fixed-size int64 counter buffer; None
specializations contain no counter writes. No clocks run inside JIT loops.
"""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from time import perf_counter

import numpy as np


COUNTERS = (
    "mf61_evaluations", "load_solves", "load_iterations", "integration_substeps",
    "actuator_pair_samples", "dense_trace_segments", "swept_queries", "aabb_rejects",
    "sat_checks", "exterior_checks", "refinement_nodes", "refinement_bisections",
    "occupancy_cells", "occupied_candidates", "event_search_passes", "pair_queries",
    "repeated_pair_queries", "physical_contact_resolutions", "predictive_stops",
)


FINISH_COUNTERS = ("active_paths", "segments", "level_tests", "hull_rejects",
                   "constant_candidates", "root_solves", "roots_returned", "event_candidates")
COPY_COUNTERS = ("arrays", "array_bytes", "containers", "memo_hits")


class SerialPhysicsProfile:
    """Exclusive host timing, matching OnPolicyTrainer.measure_region accounting.

    Inclusive envelopes are exported separately and must never be summed with
    exclusive stages. This collector needs no trainer, learner or policy network.
    """

    def __init__(self):
        self.seconds = defaultdict(float)
        self.inclusive_seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.counters = np.zeros(len(COUNTERS), dtype=np.int64)
        self.finish_counts = np.zeros(len(FINISH_COUNTERS), dtype=np.int64)
        self.copy_counts = np.zeros(len(COPY_COUNTERS), dtype=np.int64)
        self.hybrid_records = []

    def clear(self):
        self.seconds.clear()
        self.inclusive_seconds.clear()
        self.calls.clear()
        self.counters.fill(0)
        self.finish_counts.fill(0)
        self.copy_counts.fill(0)
        self.hybrid_records.clear()

    @contextmanager
    def region(self, name):
        previous = sum(self.seconds.values())
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            nested = sum(self.seconds.values()) - previous
            self.seconds[name] += max(0., elapsed - nested)
            self.inclusive_seconds[name] += elapsed
            self.calls[name] += 1

    def export(self, wall_seconds):
        return {
            "exclusive_seconds": dict(self.seconds),
            "exclusive_percent": {k: 100 * v / wall_seconds for k, v in self.seconds.items()},
            "inclusive_seconds_do_not_sum": dict(self.inclusive_seconds),
            "calls": dict(self.calls),
            "counters": dict(zip(COUNTERS, map(int, self.counters))),
            "finish_counts": dict(zip(FINISH_COUNTERS, map(int, self.finish_counts))),
            "detach_counts": dict(zip(COPY_COUNTERS, map(int, self.copy_counts))),
            "hybrid_records_separate_inclusive_detail": list(self.hybrid_records),
            "unattributed_seconds": max(0., wall_seconds - sum(self.seconds.values())),
        }

    @contextmanager
    def attach(self, simulator, *, task=None, environment=None, detail=False):
        """Use with an initialized simulator/task; no global monkeypatches."""
        with ExitStack() as resources:
            def assign(owner, name, value):
                had = name in vars(owner)
                old = vars(owner).get(name)
                setattr(owner, name, value)
                def restore():
                    if had:
                        setattr(owner, name, old)
                    else:
                        delattr(owner, name)
                resources.callback(restore)

            def instrument(owner, name, stage):
                original = getattr(owner, name, None)
                if original is None:
                    return
                @wraps(original)
                def timed(*args, **kwargs):
                    with self.region(stage):
                        return original(*args, **kwargs)
                assign(owner, name, timed)

            assign(simulator, "profile_region", self.region)
            assign(simulator, "profile_counters", self.counters)
            instrument(simulator, "step", "simulator_bookkeeping")
            instrument(simulator, "current_observation", "state_copy")
            instrument(simulator, "refresh_scans", "sensor_refresh")
            instrument(simulator.scan_simulator, "scan", "lidar_rays")
            for car in simulator.agents:
                instrument(car, "update_pose", "vehicle_bookkeeping")
                instrument(car, "compute_scan", "lidar_preparation")
                instrument(car, "check_ttc", "predictive_ttc")
                instrument(car, "ray_cast_agents", "lidar_opponents")
                instrument(car, "stop_at_contact", "contact_state")
                if car.nonlinear:
                    assign(car._physics, "profile_counters", self.counters)
                    instrument(car._physics, "advance", "rk4_integration")
            if task is not None:
                assign(task, "profile_region", self.region)
                instrument(task, "step", "task_bookkeeping")
                instrument(task, "_make_snapshot", "snapshots")
            env = task.env if task is not None else environment
            if env is not None:
                instrument(env, "step", "environment_bookkeeping")
                for name in ("_split_obs", "_attach_central_state", "get_global_state",
                             "_central_state_tensor"):
                    instrument(env, name, "observations")
                instrument(env, "_update_state", "state_copy")
                instrument(env, "get_agent_state", "snapshots")
                instrument(env, "_update_centerline_observation_facts", "frenet")
                instrument(env, "_inject_frenet_neighbors", "frenet_neighbors")
                instrument(env, "_inject_track_previews", "track_preview")
                if env._lap_tracker is not None:
                    instrument(env._lap_tracker, "update", "finish_intersections")
                if detail and env._lap_tracker is not None:
                    assign(env._lap_tracker, "profile_counts", self.finish_counts)
                    assign(env._lap_tracker, "profile_region", self.region)
            if task is not None:
                if detail:
                    assign(task, "profile_copies", self.copy_counts)
                    instrument(task, "_detach_state", "state_detach")
                for composer in task.obs_composers.values():
                    instrument(composer, "wrap", "observation_composition")
                for composer in task.reward_composers.values():
                    instrument(composer, "compute", "rewards")
                for controller in task.fixed_controllers.values():
                    instrument(controller, "act", "fixed_controller")
                    hybrid = getattr(controller, "controller", controller)
                    if detail and hasattr(hybrid, "profile_hybrid"):
                        assign(hybrid, "profile_hybrid", self.hybrid_records.append)
            yield self
