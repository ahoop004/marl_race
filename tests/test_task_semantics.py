"""Characterize existing behavior that the RaceTask extraction must preserve."""
import numpy as np
import pytest
import torch

from agents.common import compute_gae
from agents.mappo import MAPPOAgent
from env.collision_state import RaceLifecycle, apply_episode_termination_policy
from env.types import AgentRaceStatus
from training.marl_trainer import map_mappo_learning_rewards
from wrappers.actions.composer import ActionComposer
from wrappers.rewards.composer import RewardComposer


AGENTS = ("car_0", "car_1", "car_2", "car_3")
LEARNERS = list(AGENTS[:2])


@pytest.mark.parametrize("mode,done", [("any_agent", True), ("all_agents", False), ("all_trainable", False)])
def test_one_crash_obeys_episode_policy_without_crashing_survivors(mode, done):
    lifecycle = RaceLifecycle(AGENTS, target_laps=1)
    lifecycle.record_collision("car_0", step=0)
    flags, episode_done = apply_episode_termination_policy(
        {agent: agent == "car_0" for agent in AGENTS},
        dict.fromkeys(AGENTS, False),
        active_agents=AGENTS, possible_agents=AGENTS, trainable_agents=LEARNERS,
        mode=mode,
    )
    assert episode_done is done
    assert flags["car_0"]
    assert flags["car_1"] is (mode == "any_agent")
    assert lifecycle.active_agents == AGENTS[1:]
    assert lifecycle.records["car_1"].status is AgentRaceStatus.ACTIVE


@pytest.mark.parametrize("mode,done", [("all_agents", False), ("all_trainable", True)])
def test_finishing_learners_can_leave_fixed_opponents_active(mode, done):
    lifecycle = RaceLifecycle(AGENTS, target_laps=1)
    for learner in LEARNERS:
        lifecycle.record_lap_crossing(learner, step=0)
    _, episode_done = apply_episode_termination_policy(
        {agent: agent in LEARNERS for agent in AGENTS}, dict.fromkeys(AGENTS, False),
        active_agents=AGENTS, possible_agents=AGENTS, trainable_agents=LEARNERS,
        mode=mode,
    )
    assert episode_done is done
    assert lifecycle.active_agents == AGENTS[2:]
    assert not lifecycle.episode_done
    assert lifecycle.records["car_0"].finish_position == 1
    assert lifecycle.records["car_1"].finish_position == 2


def test_team_mean_keeps_configured_denominator_after_one_learner_retires():
    rewards = map_mappo_learning_rewards(
        {"car_1": 6.0}, trainable_ids=LEARNERS,
        reward_mode="team_shared", team_reward_reduction="mean",
    )
    assert rewards == {"car_1": 3.0}


def test_shared_finish_rewards_are_incremental_and_reset_for_the_next_race():
    composer = RewardComposer.from_config({"team_race_result": {
        "enabled": True, "objective": "combined",
        "rank_bonus": 60.0, "both_finish_bonus": 100.0,
    }})
    infos = {agent: {} for agent in AGENTS}
    context = {"all_infos": infos, "trainable_agent_ids": LEARNERS,
               "opponent_agent_ids": list(AGENTS[2:])}
    infos["car_0"] = {"terminal_reason": "race_complete", "finish_position": 1}
    assert composer.compute(context) == (0, {})  # Team terms stay out of local rewards.
    assert composer.compute(context, team=True) == (30.0, {"team_result/rank": 30.0})
    assert composer.compute(context, team=True) == (0, {})
    infos["car_1"] = {"terminal_reason": "race_complete", "finish_position": 2}
    reward, breakdown = composer.compute(context, team=True)
    assert reward == pytest.approx(120.0)
    assert breakdown == pytest.approx({
        "team_result/rank": 20.0, "team_result/both_finished": 100.0,
    })
    assert composer.compute(context, team=True) == (0, {})
    composer.reset()
    infos["car_1"] = {}
    assert composer.compute(context, team=True)[0] == 30.0


def test_truncation_bootstraps_final_state_without_credit_from_the_next_episode():
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 2.0]), values=torch.tensor([2.0, 1000.0]),
        terminated=torch.tensor([0.0, 1.0]), truncated=torch.tensor([1.0, 0.0]),
        next_value=999.0, gamma=0.9, gae_lambda=1.0,
        final_values=torch.tensor([3.0, float("nan")]),
    )
    torch.testing.assert_close(advantages, torch.tensor([1.7, -998.0]))
    torch.testing.assert_close(returns, torch.tensor([3.7, 2.0]))


def test_collection_cut_bootstraps_without_a_task_boundary():
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0]), values=torch.tensor([2.0]),
        terminated=torch.tensor([0.0]), truncated=torch.tensor([0.0]),
        next_value=3.0, gamma=0.9, gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([1.7]))
    torch.testing.assert_close(returns, torch.tensor([3.7]))


def test_joint_team_credit_continues_after_an_individual_learner_finishes():
    agent = MAPPOAgent(
        obs_dim=2, global_state_dim=1,
        action_low=np.full(2, -1, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32), agent_ids=LEARNERS,
        params={"n_steps": 4, "hidden_dims": [4], "device": "cpu",
                "critic_mode": "shared_team", "reward_mode": "team_shared",
                "team_return_mode": "joint", "gamma": 0.9, "gae_lambda": 1.0},
    )
    for index, ids in enumerate((LEARNERS, ["car_1"], ["car_1"])):
        terminal = index == 2
        agent.store_batch(
            ids, observations={aid: np.zeros(2, dtype=np.float32) for aid in ids},
            global_state=np.zeros(1, dtype=np.float32),
            actions={aid: np.zeros(2, dtype=np.float32) for aid in ids},
            rewards=dict.fromkeys(ids, float(index + 1)),
            log_probs=dict.fromkeys(ids, 0.0), values=dict.fromkeys(ids, 0.0),
            terminated={aid: terminal or aid == "car_0" for aid in ids},
            truncated=dict.fromkeys(ids, False),
        )
        agent.store_team_step(ids, reward=index + 1, value=0.0, terminal=terminal)
    _, returns = agent.compute_team_gae(next_value=1000.0)
    torch.testing.assert_close(returns, torch.tensor([5.23, 4.7, 3.0]))
    assert agent.buffers["car_0"].size() == 1
    assert agent.buffers["car_1"].size() == 3


def test_integrated_action_composers_have_independent_state_and_reset():
    def make_composer():
        return ActionComposer.from_config(
            np.array([-0.4, -5.0], dtype=np.float32),
            np.array([0.4, 10.0], dtype=np.float32),
            {"speed_control": "acceleration", "max_acceleration": 2.0,
             "max_deceleration": 3.0}, decision_dt=0.15,
        )

    first, second = make_composer(), make_composer()
    action = np.array([0.0, 1.0], dtype=np.float32)
    assert first.process(action)[1] == pytest.approx(0.3)
    assert first.process(action)[1] == pytest.approx(0.6)
    assert second.process(action)[1] == pytest.approx(0.3)
    first.reset()
    assert first.process(action)[1] == pytest.approx(0.3)
    np.testing.assert_array_equal(action, [0.0, 1.0])
