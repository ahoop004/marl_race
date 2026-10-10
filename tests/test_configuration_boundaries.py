"""Task construction and experiment validation have independent contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from application.configuration import validate_experiment_scenario
from core.agent_roles import resolve_agent_roles
from core.environment_config import resolve_environment_config
from core.scenario import ScenarioError, load_and_expand_scenario, validate_scenario
from core.task_builder import create_race_task
from env.spaces import SpaceSpec
from env.types import GlobalState


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def ppo_scenario():
    return load_and_expand_scenario(str(SCENARIOS / "ppo_lap_completion_pretrain.yaml"))


def test_explicit_action_ownership_does_not_require_a_supported_learner():
    scenario = ppo_scenario()
    scenario["agents"]["car_0"]["algorithm"] = "future_learner"
    scenario["agents"]["car_0"]["trainable"] = True
    validate_scenario(scenario)
    assert resolve_agent_roles(scenario["agents"]).policy_agents == ("car_0",)
    with pytest.raises(ScenarioError, match="unknown algorithm"):
        validate_experiment_scenario(scenario)


def test_legacy_scenarios_resolve_the_same_action_owners():
    scenario = load_and_expand_scenario(str(SCENARIOS / "mappo_2v2_completion_scratch.yaml"))
    expected = resolve_agent_roles(scenario["agents"])
    for config in scenario["agents"].values():
        config.pop("trainable", None)
    assert resolve_agent_roles(scenario["agents"]) == expected
    validate_experiment_scenario(scenario)


def test_cli_rollout_override_uses_resolved_legacy_action_ownership():
    from application.cli import apply_cli_overrides

    scenario = ppo_scenario()
    scenario["agents"]["car_0"].pop("trainable", None)
    args = SimpleNamespace(
        seed=None, episodes=None, wandb=False, no_wandb=False,
        render=False, no_render=False, rollout_steps_per_env=7, num_envs=2,
    )
    overridden = apply_cli_overrides(scenario, args)
    assert overridden["agents"]["car_0"]["params"]["n_steps"] == 14


@pytest.mark.parametrize("path,value,message", [
    (("mappo", "reward_mode"), "individual", "MAPPO completion requires"),
    (("experiment", "collector_scheduling"), "unsupported", "collector_scheduling"),
    (("environment", "action_repeat"), 2, "joint team returns"),
    (("evaluation", "selection_strategy"), "lap_time", "Unsupported evaluation selection"),
])
def test_experiment_restrictions_survive_validation_split(path, value, message):
    scenario = load_and_expand_scenario(str(SCENARIOS / "mappo_2v2_completion_scratch.yaml"))
    scenario.setdefault(path[0], {})[path[1]] = value
    with pytest.raises(ScenarioError, match=message):
        validate_experiment_scenario(scenario)


def test_environment_and_observations_share_one_resolved_configuration(monkeypatch):
    from core import setup, task_builder

    scenario = ppo_scenario()
    scenario["evaluation"].update(target_laps=7, max_steps=23, no_progress={"enabled": False})
    original = deepcopy(scenario)
    configs = {}
    closes = []
    bounds = SpaceSpec((2,), [-1, 0], [1, 5])

    def environment(config, agents, seed):
        configs["physics"] = config
        return SimpleNamespace(
            possible_agents=("car_0",), params={}, action_spaces={"car_0": bounds},
            timestep=.01, close=lambda: closes.append(True),
            get_global_state=lambda: GlobalState(("car_0",), np.zeros(3)),
        )

    build_observations = setup.build_obs_composers

    def observations(agents, ids, config, directory):
        configs["observations"] = config
        return build_observations(agents, ids, config, directory)

    monkeypatch.setattr(setup, "create_environment", environment)
    monkeypatch.setattr(task_builder, "build_obs_composers", observations)
    task = create_race_task(scenario, scenario_dir=SCENARIOS, mode="eval")
    try:
        assert configs["physics"] is configs["observations"]
        assert configs["physics"]["physics_phase"] == "eval"
        assert configs["physics"]["target_laps"] == 7
        assert configs["physics"]["max_steps"] == 23
        assert configs["physics"]["no_progress"] == {"enabled": False}
        assert scenario == original
    finally:
        task.close()
    assert closes == [True]


def test_task_specification_needs_no_reset_and_owns_its_data(monkeypatch):
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    task = create_race_task(ppo_scenario(), scenario_dir=SCENARIOS)
    try:
        spec = task.spec
        assert not task.agents
        spec.action_lows["car_0"].fill(1000)
        spec.observation_contracts["car_0"].clear()
        assert (task.spec.action_lows["car_0"] < 1000).all()
        assert task.spec.observation_contracts["car_0"]
    finally:
        task.close()


def test_episode_metadata_is_stable_and_detached_after_steps_and_resets(monkeypatch):
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    task = create_race_task(ppo_scenario(), scenario_dir=SCENARIOS)
    try:
        task.reset(seed=11)
        metadata = task.episode_metadata
        initial = deepcopy(metadata.spawn_configuration)
        task.step({"car_0": np.zeros(2, dtype=np.float32)})
        assert task.episode_metadata == metadata
        metadata.spawn_configuration.clear()
        assert task.episode_metadata.spawn_configuration == initial
        task.reset(seed=12)
        assert initial["initial_states"]["car_0"]
    finally:
        task.close()


def test_resolution_preserves_source_nested_configuration():
    scenario = ppo_scenario()
    original = deepcopy(scenario)
    resolved = resolve_environment_config(scenario, mode="eval", scenario_dir=SCENARIOS)
    resolved["vehicle_params"]["v_max"] = 1234
    assert scenario == original
