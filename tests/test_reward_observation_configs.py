from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from agents.common.observations import observation_layout
from core.feature_requirements import derive_environment_feature_requirements
from core.scenario import ScenarioError, load_and_expand_scenario, validate_scenario
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "scenarios").rglob("*.yaml"))


@pytest.mark.parametrize("key", ["centerline_progress", "team_race_penalties", "terminal_timeout", "progres_delta_bonus"])
@pytest.mark.parametrize("enabled", [True, False])
def test_unknown_rewards_fail_alongside_valid_reward(key, enabled):
    with pytest.raises(ValueError, match=key):
        RewardComposer.from_config({"reward": {
            "progress_delta_bonus": {"enabled": True}, key: {"enabled": enabled},
        }})


@pytest.mark.parametrize("key", ["ego_state", "prev_action", "target_frenet", "centerline", "lidarr"])
@pytest.mark.parametrize("enabled", [True, False])
def test_unknown_observations_fail_alongside_valid_observation(key, enabled):
    with pytest.raises(ValueError, match=key):
        ObservationComposer.from_config({"observation": {
            "lidar": {"enabled": True}, key: {"enabled": enabled},
        }}, {})


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda path: path.stem)
def test_checked_in_scenarios_build_composers(path):
    scenario = load_and_expand_scenario(path)
    for config in scenario["agents"].values():
        if config["algorithm"] not in {"ppo", "mappo"}:
            continue
        composer = ObservationComposer.from_config(config["observation"], scenario["environment"])
        reward = RewardComposer.from_config(config["reward"])
        assert reward is not None
        common, driving, width = observation_layout(composer.contract)
        assert driving == 158
        assert composer.obs_dim == width
        assert "frenet_neighbors" not in common["observation"]


def test_user_supplied_yaml_with_includes_and_task_metadata(tmp_path):
    (tmp_path / "reward_base.yaml").write_text("reward:\n  progress_delta_bonus:\n    enabled: true\n    weight: 3\n    positive_only: false\n")
    reward_path = tmp_path / "reward.yaml"
    reward_path.write_text("includes: [reward_base.yaml]\ntask: {name: custom}\n")
    reward = RewardComposer.from_file(str(reward_path))
    assert reward.compute({"info": {"centerline": {"progress_delta": -.25}}}) == (
        -.75, {"progress_delta/bonus": -.75})
    observation_path = tmp_path / "observation.yaml"
    observation_path.write_text("observation:\n  lidar: {enabled: true, normalize: true}\n")
    composer = ObservationComposer.from_file(str(observation_path), {"lidar_beams": 2, "lidar_range": 10})
    np.testing.assert_array_equal(composer.wrap({"lidar": [5, 20]}), [.5, 1])


def test_team_support_requests_geometry_without_neighbor_observations(tmp_path):
    requirements = derive_environment_feature_requirements({"car_0": {
        "observation": {"lidar": {"enabled": True}},
        "reward": {"team_support": {"enabled": True}},
    }}, scenario_dir=tmp_path)
    assert requirements.centerline_progress_agents == ("car_0",)
    assert requirements.frenet_neighbor_agents == ("car_0",)
    assert requirements.track_preview_agents == ()


def test_driving_prefix_is_preserved_for_team_transfer():
    solo = load_and_expand_scenario(ROOT / "scenarios/ppo_lap_completion_pretrain.yaml")
    team = load_and_expand_scenario(ROOT / "scenarios/mappo_2v2_completion_scratch.yaml")
    source = ObservationComposer.from_config(solo["agents"]["car_0"]["observation"], solo["environment"])
    destination = ObservationComposer.from_config(team["agents"]["car_0"]["observation"], team["environment"])
    original = deepcopy(destination.contract)
    source_common, source_driving, source_width = observation_layout(source.contract)
    dest_common, dest_driving, dest_width = observation_layout(destination.contract)
    assert source_common == dest_common
    assert source_driving == dest_driving == source_width == 158
    assert dest_width == 192
    assert destination.contract == original


def test_unsupported_adapter_transfer_fails_explicitly():
    scenario = load_and_expand_scenario(ROOT / "scenarios/mappo_2v2_completion_scratch.yaml")
    scenario["training_defaults"]["adapter_transfer"] = {"checkpoint": "old.pt"}
    with pytest.raises(ScenarioError, match="adapter_transfer is no longer supported"):
        validate_scenario(scenario)


@pytest.mark.parametrize("mode", ["independent", "shared", "lora"])
def test_ppo_actor_transfer_preserves_driving_policy(tmp_path, mode):
    import torch
    from agents.mappo import MAPPOAgent
    from agents.ppo import PPOAgent

    solo = load_and_expand_scenario(ROOT / "scenarios/ppo_lap_completion_pretrain.yaml")
    team = load_and_expand_scenario(ROOT / "scenarios/mappo_2v2_completion_scratch.yaml")
    source_obs = ObservationComposer.from_config(solo["agents"]["car_0"]["observation"], solo["environment"])
    dest_obs = ObservationComposer.from_config(team["agents"]["car_0"]["observation"], team["environment"])
    low, high = np.array([-1, -1]), np.array([1, 1])
    params = {"pi_hidden_dims": [8], "vf_hidden_dims": [8], "n_steps": 2, "device": "cpu"}
    source = PPOAgent(source_obs.obs_dim, low, high,
                      {**params, "_observation_contract": source_obs.contract})
    path = tmp_path / "ppo.pt"
    source.save(str(path))
    dest_params = {**params, "actor_mode": "shared" if mode == "lora" else mode,
                   "_observation_contract": dest_obs.contract,
                   "pretrained_actor_observation_extension": "frenet_neighbors"}
    if mode == "lora":
        dest_params["lora"] = {"mode": "per_agent", "rank": 2, "alpha": 2,
                               "train_log_std": True, "per_agent_log_std": True}
    destination = MAPPOAgent(dest_obs.obs_dim, 10, low, high, ["car_0", "car_1"], dest_params)
    critic_before = deepcopy(destination.critic.state_dict())
    destination.load_pretrained_actor(str(path))
    driving = np.linspace(-1, 1, source_obs.obs_dim, dtype=np.float32)
    # Nonzero traffic inputs must have no effect immediately after transfer.
    traffic = np.concatenate([driving, np.ones(dest_obs.obs_dim - source_obs.obs_dim, dtype=np.float32)])
    expected = source.act(driving, deterministic=True)[0]
    for aid in destination.agent_ids:
        actual, _ = destination.act(traffic, deterministic=True, agent_id=aid)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    for key, value in critic_before.items():
        assert torch.equal(destination.critic.state_dict()[key], value)
