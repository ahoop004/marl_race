"""Matched P2 replay and detailed instrumentation preserve the physics contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from physics.profiling import SerialPhysicsProfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("physics_p2", ROOT / "scripts/benchmark_physics_p2.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)


@pytest.mark.parametrize("map_id", ["circle_map", "Budapest_map"])
def test_matched_replay_and_detail_preserve_full_outputs(map_id):
    task, drivers, reset, _ = p2.build_workload(map_id, 12)
    try:
        frames, samples = p2.record_trace(task, drivers, reset, 12)
        assert len(frames) == 12
        assert samples.shape == (16, 8)
        for layer in p2.LAYERS:
            row = p2.replay(task, reset, frames, layer, capture=True)
            assert row["exact_physics_parity"]
        for layer in ("environment", "task_replay", "task_live"):
            profile = SerialPhysicsProfile()
            row = p2.replay(task, reset, frames, layer, profile=profile, capture=True)
            assert row["exact_physics_parity"]
            assert sum(profile.seconds.values()) <= row["wall_seconds"]
            assert profile.finish_counts[1] == 12 * 4 * 50
            assert profile.finish_counts[2] == 2 * profile.finish_counts[1]
            assert profile.finish_counts[3] <= profile.finish_counts[2]
            assert "profile_counts" not in vars(task.env._lap_tracker)
            assert task.profile_copies is None
            assert all(adapter.controller.profile_hybrid is None for adapter in task.fixed_controllers.values())
            if layer == "task_live":
                assert len(profile.hybrid_records) == 24
                for record in profile.hybrid_records:
                    assert sum(record["stages"].values()) == pytest.approx(record["seconds"])
                assert profile.copy_counts[0] > 0 and profile.copy_counts[1] > 0
            elif layer == "task_replay":
                assert not profile.hybrid_records
                assert profile.copy_counts[0] > 0
        # Sampling exact force queries is independent of normal integration.
        result = p2.solver_breakdown(samples[:4], task.env.sim.agents[0]._physics._dynamics_params,
                                     repeats=1, batches=1)
        assert result["exact_force_trace_parity"]
        assert result["mf61_calls"] == 2 * result["force_calls"]
    finally:
        task.close()


@pytest.mark.parametrize("finish_x", [2.05, 2.3])
def test_finish_root_counts_preserve_finish_collision_chronology(tmp_path, finish_x):
    from test_swept_collision import finish_env
    params = yaml.safe_load((ROOT / "configs/vehicle/combined_slip.yaml").read_text())
    results = []
    for enabled in (False, True):
        env = finish_env(tmp_path, params, finish_x=finish_x)
        try:
            profile = SerialPhysicsProfile()
            if enabled:
                with profile.attach(env.sim, environment=env, detail=True):
                    output = env.step({"car_0": np.array([0., 100.])})
                assert profile.finish_counts[2] > 0
                if finish_x == 2.05:
                    assert profile.finish_counts[5] > 0
                    assert profile.calls["finish_polyroots"] == profile.finish_counts[5]
            else:
                output = env.step({"car_0": np.array([0., 100.])})
            results.append(p2.p1.digest((output, env.sim.collision_times,
                                        env.sim.agents[0].physics_state)).hexdigest())
            assert output[2]["car_0"]
            assert output[4]["car_0"]["terminal_reason"] == ("race_complete" if finish_x == 2.05 else "collision")
        finally:
            env.close()
    assert results[0] == results[1]


def test_detach_copy_counts_respect_alias_memo_and_readonly_arrays():
    from tasks.race_task import _detach
    array = np.arange(8, dtype=np.float64)
    array.setflags(write=False)
    counts = np.zeros(4, dtype=np.int64)
    source = {"left": array, "right": array}
    copied = _detach(source, counters=counts)
    assert copied["left"] is copied["right"] and copied["left"] is not array
    assert not copied["left"].flags.writeable
    np.testing.assert_array_equal(counts, [1, array.nbytes, 1, 1])
