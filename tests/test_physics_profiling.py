"""P1 instrumentation must preserve the committed simulation contract exactly."""
from pathlib import Path

import numpy as np
import pytest

from physics.profiling import COUNTERS, SerialPhysicsProfile

# Use the same workload builders as the serial benchmark, without a learner.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "benchmark_physics", Path(__file__).resolve().parents[1] / "scripts/benchmark_physics.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("name", [f"B{i}" for i in range(1, 7)])
def test_profiling_rollout_is_bit_identical_and_restores_instances(tmp_path, name, monkeypatch):
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    sim, task, reset, advance, _ = bench.build_case(name, bench.synthetic_map(tmp_path), bench.ROOT, 12)
    profile = SerialPhysicsProfile()
    try:
        with monkeypatch.context() as disabled:
            disabled.setattr("physics.profiling.perf_counter",
                             lambda: pytest.fail("disabled profiling read a clock"))
            plain = bench.run_rollout(sim, task, reset, advance, 12, capture=True)
        original_sim = vars(sim).copy()
        original_car = vars(sim.agents[0]).copy()
        measured = bench.run_rollout(sim, task, reset, advance, 12, profile, capture=True)
        assert measured["digest"] == plain["digest"]
        assert vars(sim).keys() == original_sim.keys()
        assert vars(sim.agents[0]).keys() == original_car.keys()
        assert sim.profile_region is None and sim.profile_counters is None
        assert "advance" not in vars(sim.agents[0]._physics)
        counts = measured["profile"]["counters"]
        assert counts["integration_substeps"] == measured["vehicle_steps"] * 50
        stops = counts["physical_contact_resolutions"] + counts["predictive_stops"]
        assert counts["load_solves"] == measured["vehicle_steps"] * 201 + stops
        assert counts["actuator_pair_samples"] == measured["vehicle_steps"] * 151 + stops
        assert counts["mf61_evaluations"] == 2 * (counts["load_iterations"] + 2 * counts["load_solves"])
        assert sum(measured["profile"]["exclusive_seconds"].values()) <= measured["wall_seconds"]
        if name in {"B4", "B5"}:
            assert counts["physical_contact_resolutions"] > 0
            assert counts["refinement_bisections"] > 0
        if name == "B5":
            assert counts["repeated_pair_queries"] > 0
        if task:
            assert measured["profile"]["calls"]["rewards"] > 0
            assert measured["profile"]["calls"]["finish_intersections"] > 0
            assert task.episode_done  # truncation/terminal masks included in digest
        with pytest.raises(RuntimeError, match="sentinel"):
            with profile.attach(sim, task=task):
                raise RuntimeError("sentinel")
        assert sim.profile_region is None and sim.profile_counters is None
    finally:
        if task:
            task.close()


def test_nested_region_accounting_is_exclusive(monkeypatch):
    clock = iter([0., 1., 3., 5.])
    monkeypatch.setattr("physics.profiling.perf_counter", lambda: next(clock))
    profile = SerialPhysicsProfile()
    with profile.region("outer"):
        with profile.region("inner"):
            pass
    assert profile.seconds == {"inner": 2., "outer": 3.}
    assert profile.inclusive_seconds == {"inner": 2., "outer": 5.}
    profile.clear()
    assert not profile.seconds and not profile.calls
    np.testing.assert_array_equal(profile.counters, np.zeros(len(COUNTERS)))
