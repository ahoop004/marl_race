"""Hybrid timing is opt-in and does not change actions or optimizer results."""
from contextlib import ExitStack
from copy import deepcopy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.fixed.hybrid import HybridPPFTGAgent
from core.scenario import ScenarioError, load_and_expand_scenario
from training.configuration import validate_training_scenario
from training.hooks import CSVHook
from loggers.csv_logger import CSVLogger
from wrappers.actions.composer import WheelReferenceAdapter
from test_hybrid_controller import fixture
from test_torchrl_ppo import Capture, trainer


def test_optional_timers_preserve_actions_and_count_conditional_stages(fixture, monkeypatch):
    f = fixture
    plain = HybridPPFTGAgent(dict(agent_id="car_0", max_speed=8.))
    plain.set_env(f.env)
    records = []
    measured = f.controller
    measured.profile_hybrid = records.append
    clock = itertools.count(step=.001)
    for obstructed in (False, True, True, False, False, False, False, False):
        f.obs["lidar"][:] = 10.
        if obstructed:
            f.obs["lidar"][abs(f.angles) < .1] = 3.
        monkeypatch.setattr("agents.fixed.hybrid.perf_counter", lambda: pytest.fail("disabled timer read the clock"))
        expected = plain.act(f.obs)
        monkeypatch.setattr("agents.fixed.hybrid.perf_counter", lambda: next(clock))
        np.testing.assert_array_equal(expected, measured.act(f.obs))
    assert [row["mode"] for row in records] == ["pp", "ftg", "ftg", "ftg", "ftg", "ftg", "ftg", "pp"]
    assert sum(row["mode_switch"] for row in records) == 2
    for row in records:
        assert row["seconds"] == pytest.approx(sum(row["stages"].values()))
        assert row["stage_calls"].get("gap_selection", 0) == (row["mode"] == "ftg")
        assert row["stage_calls"]["pure_pursuit"] == 1
    assert [row["cold_start"] for row in records] == [1] + [0] * 7
    measured.reset()
    assert measured.profile_hybrid == records.append
    measured.act(f.obs)
    assert records[-1]["cold_start"] == 1
    f.env.centerline_points = f.env.centerline_points + [0., .1]
    measured.act(f.obs)
    assert records[-1]["cold_start"] == records[-1]["map_changed"] == 1


@pytest.mark.parametrize("case", ["pose", "velocity", "yaw", "steering", "grip", "missing_scan", "no_gap", "emergency"])
def test_safety_and_early_returns_are_profiled(fixture, case):
    f = fixture
    records = []
    f.controller.profile_hybrid = records.append
    if case in {"pose", "velocity"}:
        f.obs[case][0] = np.nan
    elif case in {"yaw", "steering"}:
        f.obs["angular_velocity" if case == "yaw" else "steering_angle"] = np.nan
    elif case == "grip":
        f.env.get_global_state = lambda: SimpleNamespace(metadata={"physics": {"mu": np.nan}})
    elif case == "missing_scan":
        f.obs.pop("lidar")
    elif case == "no_gap":
        f.obs["lidar"][:] = 0.
    else:
        f.env.possible_agents.append("wreck")
        f.obs["velocity"][:] = [2., 0.]
        f.env.get_agent_state = lambda aid: SimpleNamespace(pose=np.array([2., 0., 0.]), velocity=np.zeros(2))
    assert f.controller.act(f.obs)[1] == 0.
    row, = records
    assert row["stopped"] == 1
    assert row["seconds"] == pytest.approx(sum(row["stages"].values()))
    if case in {"pose", "velocity", "yaw", "steering", "grip"}:
        assert row["invalid_state"] == 1 and row["mode"] == "invalid"
    else:
        assert row[{"emergency": "emergency_brake"}.get(case, case)] == 1
    json.dumps(row, allow_nan=False)


def test_aggregation_tracks_retirement_and_exclusive_stages(fixture):
    training, _ = trainer(Capture(False))
    training.measure_hybrid = training.measure_stages = True
    training.trainable_ids = ["car_0", "car_1"]
    f = fixture
    f.env.possible_agents = ["car_2", "car_3"]
    f.env.sim.agents *= 2
    f.env.get_agent_state = lambda aid: SimpleNamespace(pose=np.array([200., 0., 0.]), velocity=np.zeros(2))
    controllers = {aid: HybridPPFTGAgent(dict(agent_id=aid)) for aid in f.env.possible_agents}
    for controller in controllers.values():
        controller.set_env(f.env)
    training.task.fixed_controllers = {aid: WheelReferenceAdapter(controller, f.env.params["wheel_actuators"])
                                       for aid, controller in controllers.items()}
    previous_callback = lambda record: None
    controllers["car_2"].profile_hybrid = previous_callback
    with ExitStack() as resources:
        training._instrument_hybrid(resources)
        training._timing_phase = "warmup"
        for active in (("car_0", "car_1", "car_2", "car_3"), ("car_1", "car_2", "car_3"),
                       ("car_2", "car_3"), ("car_3",)):
            training.task.env.agents = list(active)
            for aid in active:
                if aid in training.task.fixed_controllers:
                    training.measure_stage("fixed_controller", training.task.fixed_controllers[aid].act, f.obs)
            training.hybrid_environment_decisions["warmup"] = training.hybrid_environment_decisions.get("warmup", 0) + 1
        summary = training.hybrid_summary()
    assert controllers["car_2"].profile_hybrid is previous_callback
    assert controllers["car_3"].profile_hybrid is None
    overall = summary["overall"]
    assert overall["calls"] == training.stage_calls["warmup"]["fixed_controller"] == 7
    assert overall["environment_decisions"] == 4
    assert overall["seconds"] <= training.stage_seconds["warmup"]["fixed_controller"]
    assert overall["seconds"] == pytest.approx(sum(row["seconds"] for row in overall["stages"].values()))
    assert overall["max_ms_per_call"] >= overall["mean_ms_per_call"] > 0.
    assert summary["agent"]["car_2"]["calls"] == 3
    assert summary["agent"]["car_3"]["calls"] == 4
    assert {key: row["calls"] for key, row in summary["active_vehicles"].items()} == {"4": 2, "3": 2, "2": 2, "1": 1}
    assert summary["active_learners"]["0"]["calls"] == 3
    assert summary["active_hybrid"]["1"]["calls"] == 1
    assert summary["mode"]["pp"]["fraction"] == 1.
    assert summary["overall"]["stages"]["gap_selection"]["calls"] == 0
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("warmup_updates", [0, 1])
def test_phase_boundaries_reset_and_export_without_stage_timing(tmp_path, warmup_updates):
    training, _ = trainer(Capture(False))
    training.measure_hybrid = True
    training.profiling_warmup_updates = warmup_updates
    training.hooks.append(CSVHook(CSVLogger(tmp_path)))
    training.train(total_steps=9)
    assert training.hybrid_environment_decisions == ({"warmup": 4, "steady": 5} if warmup_updates else {"steady": 9})
    assert training.hybrid_summary()["overall"]["calls"] == 0  # Fixture has a different controller.
    assert json.loads((tmp_path / "run_summary.json").read_text())["hybrid_profile"]["overall"]["environment_decisions"] == 9
    training.hooks.clear()
    training.train(total_steps=12)
    assert sum(training.hybrid_environment_decisions.values()) == 3
    assert training.hybrid_totals == {}


def test_hybrid_instrumentation_preserves_mappo_optimizer_results(fixture, tmp_path):
    from test_native_torchrl_env import team_task
    from test_torchrl_mappo import params, BOUNDS, IDS
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from training.on_policy import OnPolicyTrainer

    def make():
        task = team_task({3: (("car_0", "car_1", "car_2", "car_3"), True,
                              ("car_0", "car_1", "car_2", "car_3"), ())})
        policy = TorchRLMAPPOAgent(1, task.state_space().shape[0], *BOUNDS, IDS, params(n_steps=4))
        training = OnPolicyTrainer(task, policy, hooks=[Capture(False)])
        for aid in task.fixed_controllers:
            env = deepcopy(fixture.env)
            env.possible_agents = [aid]
            controller = HybridPPFTGAgent(dict(agent_id=aid))
            controller.set_env(env)
            act = controller.act
            controller.act = lambda obs, act=act: act(fixture.obs)
            task.fixed_controllers[aid] = WheelReferenceAdapter(controller, env.params["wheel_actuators"])
        return training

    plain, measured = make(), make()
    measured.measure_hybrid = True
    measured.hooks.append(CSVHook(CSVLogger(tmp_path)))
    for module in ("actor", "critic"):
        getattr(measured.agent, module).load_state_dict(getattr(plain.agent, module).state_dict())
    rng = torch.random.get_rng_state()
    plain.train(total_steps=9)
    torch.random.set_rng_state(rng)
    measured.train(total_steps=9)
    assert plain.checkpoint_progress() == measured.checkpoint_progress()
    for module in ("actor", "critic"):
        for key, value in getattr(plain.agent, module).state_dict().items():
            torch.testing.assert_close(value, getattr(measured.agent, module).state_dict()[key], rtol=0, atol=0)
    summary = measured.hybrid_summary()
    assert summary["overall"]["calls"] == 18
    assert summary["phase"]["warmup"]["calls"] == 4
    assert summary["phase"]["steady"]["calls"] == 14
    assert sum(row["calls"] for row in summary["mode"].values()) == 18
    assert all(adapter.controller.profile_hybrid is None for adapter in measured.task.fixed_controllers.values())
    assert json.loads((tmp_path / "run_summary.json").read_text())["hybrid_profile"]["overall"]["calls"] == 18
    assert "perf/hybrid/all/stages/pure_pursuit/mean_ms_per_call" in (tmp_path / "update_metrics.csv").read_text()


def test_hybrid_serial_preset_uses_shared_opponents_and_opt_in_metrics():
    production = load_and_expand_scenario("scenarios/mappo_2v2_completion_scratch.yaml")
    reference = load_and_expand_scenario("scenarios/mappo_serial_profile.yaml")
    profile = load_and_expand_scenario("scenarios/mappo_hybrid_serial_profile.yaml")
    validate_training_scenario(profile)
    assert not production["experiment"]["measure_hybrid"]
    assert not reference["experiment"]["measure_hybrid"]
    assert reference["controllers"]["name"] == "racing_mpc"
    assert profile["experiment"]["measure_hybrid"] and profile["experiment"]["measure_stages"]
    for aid in ("car_2", "car_3"):
        assert profile["agents"][aid] == production["agents"][aid]
    assert profile["experiment"]["total_steps"] == 16384
    assert profile["experiment"]["profiling_warmup_updates"] == 1
    assert profile["training_defaults"] == reference["training_defaults"]
    for value in ("1", "null", "yes"):
        invalid = load_and_expand_scenario("scenarios/mappo_hybrid_serial_profile.yaml", overrides=[
            f"experiment.measure_hybrid={value}"])
        with pytest.raises(ScenarioError, match="measure_hybrid must be boolean"):
            validate_training_scenario(invalid)
