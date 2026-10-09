"""Check declared spaces against real racing observations."""
from pathlib import Path

import numpy as np
import pytest

from core.env_builder import create_environment
from core.map_selection import apply_map_split
from core.scenario import load_and_expand_scenario


ROOT = Path(__file__).resolve().parents[1]
ALL_SENSORS = (
    "lidar", "pose", "velocity", "acceleration", "angular_velocity",
    "target_pose", "target_collision", "lap", "collision",
)


@pytest.fixture(params=["legacy_st", "combined_slip_st"])
def race_env(request, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    scenario = load_and_expand_scenario(
        str(ROOT / "scenarios/mappo_2v2_completion_scratch.yaml")
    )
    config = apply_map_split(scenario["environment"], scenario["experiment"], "train")
    if request.param == "legacy_st":
        config["vehicle_params"] = {"model": "legacy_st"}
        config.pop("friction", None)
    config["max_steps"] = 3
    env = create_environment(config, scenario["agents"], seed=42)
    try:
        yield env
    finally:
        env.close()


def assert_observation_contract(env, observations):
    for agent, observation in observations.items():
        space = env.observation_space(agent)
        assert space is env.observation_spaces[agent]
        assert observation.keys() == space.spaces.keys()
        for key, spec in space.spaces.items():
            value = np.asarray(observation[key])
            assert value.shape == spec.shape, key
            assert value.dtype == spec.dtype, key
            assert np.isfinite(value).all(), key
            assert (value >= spec.low).all(), key
            assert (value <= spec.high).all(), key
        if "lidar" in observation:
            assert observation["lidar"] is observation["scans"]
        np.testing.assert_array_equal(observation["state"], env.get_global_state().vector)


@pytest.mark.parametrize("sensors", [None, ALL_SENSORS, ()], ids=["default", "all", "none"])
def test_reset_and_step_observation_contract(race_env, sensors):
    env = race_env
    if sensors is not None:
        env._agent_sensor_spec = {agent: sensors for agent in env.possible_agents}
        pose_space = env.observation_space("car_0").spaces["pose"]
        env._build_observation_spaces(
            pose_space.low[0], pose_space.high[0],
            pose_space.low[1], pose_space.high[1],
        )
    env.configure_agent_targets({"car_0": "car_1"})
    observations, _ = env.reset(seed=42)
    assert_observation_contract(env, observations)
    for _ in range(3):
        actions = {
            agent: np.asarray([0.05, 1.0], dtype=np.float32)
            for agent in env.agents
        }
        observations, _, _, truncations, _ = env.step(actions)
        assert_observation_contract(env, observations)
        for agent in actions:
            observation = observations[agent]
            assert observation["steering_reference"] == actions[agent][0]
            speed_key = "wheel_speed_reference" if env.sim.agents[0].nonlinear else "speed_reference"
            assert observation[speed_key] == actions[agent][1]
    assert not env.agents
    assert all(truncations.values())
    # The newly declared scalar spaces also support sampling.
    sample = env.observation_space("car_0").spaces["steering_reference"].sample()
    assert sample.shape == ()
    assert sample.dtype == np.float32


def test_lidar_alias_after_terminal_vehicle_removal(race_env):
    env = race_env
    # Use the supported terminal-car removal path, which refreshes scans
    # after observation assembly when a collision removes an obstacle.
    from env.state_buffer import TerminalAgentConfig

    env._terminal_controller.config = TerminalAgentConfig.from_mapping({
        "remove_after_clearance": True,
        "finish_clearance_steps": 0,
        "crash_clearance_steps": 0,
    })
    env.reset(seed=42)
    poses = np.stack([env.get_agent_state(agent).pose for agent in env.possible_agents])
    poses[2] = poses[0]
    env.reset(seed=42, options={"poses": poses})
    observations, _, terminations, _, _ = env.step({
        agent: np.zeros(2, dtype=np.float32) for agent in env.agents
    })
    assert terminations["car_0"] and terminations["car_2"]
    assert not env.sim.collidable_mask[0]
    assert_observation_contract(env, observations)
