from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from agents.fixed.hybrid import HybridPPFTGAgent
from core.agent_builder import build_fixed_policy_agents, create_fixed_controller
from core.scenario import load_and_expand_scenario


@pytest.fixture
def fixture():
    angles = np.linspace(-2.35, 2.35, 108)
    actuators = dict(steering_min=-.4189, steering_max=.4189, wheel_speed_min=0.,
                     wheel_speed_max=160., wheel_radius=.05, steering_time_constant=0.)
    env = SimpleNamespace(
        params=dict(model="combined_slip_st", lf=.16, lr=.17, width=.31,
                    length=.58, mu=1., wheel_actuators=actuators),
        timestep=.05, lidar_range=10., lidar_dist=0., possible_agents=["car_0"],
        centerline_points=np.column_stack((np.linspace(0, 100, 501), np.zeros(501))),
        sim=SimpleNamespace(agents=[SimpleNamespace(scan_angles=angles)]),
        get_global_state=lambda: SimpleNamespace(metadata={"physics": {"mu": 1.}}))
    controller = create_fixed_controller("hybrid_pp_ftg", dict(agent_id="car_0", max_speed=8.))
    controller.set_env(env)
    obs = dict(pose=np.array([1., 0., 0.]), velocity=np.array([3., 0.]),
               angular_velocity=0., steering_angle=0., lidar=np.full(108, 10.))
    return SimpleNamespace(controller=controller, env=env, obs=obs, angles=angles)


def test_clear_track_reaches_speed_limit_and_ignores_nearby_side_walls(fixture):
    f = fixture
    # Narrow side walls are near the ego but do not obstruct its planned path.
    side = abs(np.sin(f.angles)) > .15
    f.obs["lidar"][side] = np.minimum(10., 1. / abs(np.sin(f.angles[side])))
    for _ in range(50):
        action = f.controller.act(f.obs)
    assert action.dtype == np.float32 and action.shape == (2,)
    assert action[0] == pytest.approx(0.)
    assert action[1] == pytest.approx(8.)
    assert f.controller.mode == "pp"


def test_obstruction_activates_gap_and_clear_path_requires_hysteresis(fixture):
    f = fixture
    f.obs["lidar"][abs(f.angles) < .1] = 3.
    action = f.controller.act(f.obs)
    assert f.controller.mode == "ftg"
    assert abs(action[0]) > .01 and 0 < action[1] <= 3.15
    side = np.sign(action[0])
    for _ in range(4):
        assert np.sign(f.controller.act(f.obs)[0]) == side
    f.obs["lidar"][:] = 10.
    for _ in range(f.controller.clear_steps - 1):
        f.controller.act(f.obs)
        assert f.controller.mode == "ftg"
    f.controller.act(f.obs)
    assert f.controller.mode == "pp"


@pytest.mark.parametrize("scan", [np.zeros(108), np.full(108, np.nan), np.full(108, -np.inf), []])
def test_no_safe_or_known_path_brakes_immediately(fixture, scan):
    f = fixture
    f.controller.act(f.obs)
    f.obs["lidar"] = scan
    action = f.controller.act(f.obs)
    assert np.isfinite(action).all() and action[1] == 0.
    assert f.controller.mode == "ftg"


def test_reset_map_switch_and_invalid_pose(fixture):
    f = fixture
    f.obs["lidar"][abs(f.angles) < .1] = 3.
    first = f.controller.act(f.obs)
    for _ in range(10):
        f.controller.act(f.obs)
    f.controller.reset()
    np.testing.assert_array_equal(first, f.controller.act(f.obs))
    f.env.centerline_points = f.env.centerline_points + [0., 1.]
    f.obs["lidar"][:] = 10.
    f.controller.act(f.obs)
    assert f.controller._map_source is f.env.centerline_points
    f.obs["pose"][0] = np.nan
    np.testing.assert_array_equal(f.controller.act(f.obs), np.zeros(2, np.float32))


def test_open_path_endpoint_stops_and_low_grip_slows(fixture):
    f = fixture
    f.obs["pose"][0] = 100.
    f.obs["velocity"][:] = 0.
    assert f.controller.act(f.obs)[1] == 0.
    f.controller.reset()
    f.obs["pose"][0] = 1.
    f.env.get_global_state = lambda: SimpleNamespace(metadata={"physics": {"mu": 0.}})
    assert f.controller.act(f.obs)[1] == 0.


def test_existing_adapter_converts_once_and_respects_lower_limit(fixture):
    f = fixture
    controllers = build_fixed_policy_agents({"car_0": dict(algorithm="hybrid_pp_ftg", trainable=False,
        action_adapter="rolling_speed_to_wheel_v1", params=dict(max_speed=4.))}, vehicle_params=f.env.params)
    adapter = controllers["car_0"]
    adapter.set_env(f.env)
    physical = adapter.controller.act(deepcopy(f.obs))
    adapter.reset()
    converted = adapter.act(f.obs)
    assert converted[0] == physical[0]
    assert converted[1] == pytest.approx(physical[1] / .05)
    for _ in range(30):
        converted = adapter.act(f.obs)
    assert converted[1] == pytest.approx(80.)


@pytest.mark.parametrize("config", [dict(max_speed=-1), dict(braking=0), dict(margin=np.nan),
                                     dict(clear_steps=1.5), dict(lookahead_max=.1), dict(typo=1)])
def test_bad_config_fails_at_construction(config):
    with pytest.raises(ValueError, match="hybrid_pp_ftg"):
        HybridPPFTGAgent(config)


def test_experiment_arms_share_opponents_and_mpc_remains_selectable():
    arms = [load_and_expand_scenario(f"scenarios/mappo_2v2_completion_{arm}.yaml")
            for arm in ("scratch", "pretrained", "pretrained_shared", "lora")]
    for config in arms:
        for aid in ("car_2", "car_3"):
            assert config["agents"][aid] == arms[0]["agents"][aid]
            assert config["agents"][aid]["algorithm"] == "hybrid_pp_ftg"
            assert config["agents"][aid]["params"]["max_speed"] == 8.
    reference = load_and_expand_scenario("scenarios/mappo_2v2_completion_scratch.yaml",
                                         overrides=["controllers=racing_mpc"])
    assert reference["agents"]["car_2"]["algorithm"] == "racing_mpc"
    assert reference["agents"]["car_3"]["params"] == reference["controllers"]["racing_mpc"]
    lower = load_and_expand_scenario("scenarios/mappo_2v2_completion_scratch.yaml", overrides=[
        "controllers.hybrid_pp_ftg.max_speed=4."])
    assert lower["environment"]["max_speed"] == 8.
    assert lower["agents"]["car_2"]["params"]["max_speed"] == 4.


def test_time_aligned_emergency_brakes_for_closing_car_but_allows_safe_pass(fixture):
    f = fixture
    f.env.possible_agents.append("wreck")
    state = SimpleNamespace(pose=np.array([2., 0., 0.]), velocity=np.zeros(2))
    f.env.get_agent_state = lambda aid: state
    # The neighbor need not be active: stationary retired cars remain obstacles.
    f.env.agents = ["car_0"]
    f.obs["velocity"][:] = [2., 0.]
    assert f.controller.act(f.obs)[1] == 0.
    assert f.controller.last_plan["emergency_brake"]
    state.pose[1] = 1.
    f.controller.reset()
    assert f.controller.act(f.obs)[1] > 0.
    assert not f.controller.last_plan["emergency_brake"]


@pytest.mark.parametrize("map_id", ["circle_map", "Budapest_map"])
@pytest.mark.parametrize("max_speed", [5., 8.])
def test_real_simulation_completes_lap_and_avoids_stationary_vehicle(map_id, max_speed, monkeypatch):
    from application.benchmark_controllers import run_case
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    for case in ("solo", "stationary"):
        result = run_case("hybrid_pp_ftg", map_id, 42, case, max_speed=max_speed, max_steps=4500)
        ego = result["agents"]["car_0"]
        assert not ego["controller_errors"]
        assert ego["completed"] and ego["collisions"] == 0
        assert ego["mean_speed_mps"] > 3.
        assert ego["ftg_fraction"] == 0 if case == "solo" else ego["ftg_fraction"] > 0
