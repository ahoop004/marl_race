"""Characterize existing behavior that the RaceTask extraction must preserve."""
import numpy as np
import pytest
import torch

from env.collision_state import RaceLifecycle, apply_episode_termination_policy
from env.types import AgentRaceStatus
from adapters.rewards import RewardMapping
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
    rewards = RewardMapping("team_shared", "mean").map({"car_1": 6.0}, LEARNERS)
    assert rewards == {"car_1": 3.0}


@pytest.mark.parametrize("events,progress,expected", [
    ({}, -.01, {"progress_delta/bonus": -.01}),
    ({}, .1, {"progress_delta/bonus": .025}),
    ({"collision_event": True}, .01, {"progress_delta/collision": -2.}),
    ({"boundary_event": True}, .01, {"progress_delta/boundary": -.05}),
    ({"lap_crossed": True, "race_completed": True, "terminal_reason": "race_complete"},
     .01, {"progress_delta/bonus": .01, "lap_completion/bonus": 2.}),
    ({"lap_crossed": True, "race_completed": True, "collision_event": True},
     .01, {"progress_delta/collision": -2.}),
    ({"lap_crossed": True, "race_completed": True, "track_limits": {"exceeded": True}},
     .01, {"progress_delta/boundary": -.05}),
    ({"lap_crossed": True, "race_completed": True, "terminal_reason": "time_limit"},
     .01, {"progress_delta/bonus": .01, "timeout/penalty": -1.}),
])
def test_solo_transfer_and_team_local_completion_rewards_agree(events, progress, expected):
    from core.scenario import load_yaml_config
    configs = [load_yaml_config(f"configs/reward/{name}.yaml") for name in (
        "solo_completion", "solo_transfer", "team_completion")]
    assert configs[2]["reward"]["team_race_result"]["both_finish_bonus"] == 2.
    context = dict(timestep=.05, track_length=100., info={
        "centerline": {"progress_delta": progress}, "track_limits": {"exceeded": False}, **events})
    expected = {**expected, "step_time/penalty": -.00025 * .05}
    for config in configs:
        reward, breakdown = RewardComposer.from_config(config).compute(context)
        assert breakdown == pytest.approx(expected)
        assert reward == pytest.approx(sum(expected.values()))


@pytest.mark.parametrize("events,expected", [
    ({}, {"progress_delta/bonus": .01}),
    ({"track_limits": {"exceeded": True}}, {"progress_delta/boundary": -.05}),
    ({"boundary_event": True}, {"progress_delta/boundary": -.05}),
    ({"terminal_reason": "collision"}, {"progress_delta/collision": -2.}),
    ({"collision": True, "collision_event": True}, {"progress_delta/bonus": .01}),
    ({"lap_crossed": True, "race_completed": True},
     {"progress_delta/bonus": .01, "lap_completion/bonus": 2.}),
    ({"lap_crossed": True, "race_completed": True, "track_limits": {"exceeded": True}},
     {"progress_delta/boundary": -.05, "lap_completion/bonus": 2.}),
    ({"lap_crossed": True, "race_completed": True, "collision": True, "collision_event": False},
     {"progress_delta/bonus": .01}),
    ({"lap_crossed": True, "race_completed": True, "collision": False, "collision_event": True},
     {"progress_delta/bonus": .01, "lap_completion/bonus": 2.}),
    ({"lap_crossed": True, "race_completed": True, "terminal_reason": "collision"},
     {"progress_delta/collision": -2., "lap_completion/bonus": 2.}),
    ({"lap_crossed": True, "race_completed": True, "terminal_reason": "time_limit"},
     {"progress_delta/bonus": .01, "timeout/penalty": -1., "lap_completion/bonus": 2.}),
    ({"terminal_reason": "time_limit"}, {"progress_delta/bonus": .01, "timeout/penalty": -1.}),
    ({"lap_crossed": False, "race_completed": True}, {"progress_delta/bonus": .01}),
    ({"lap_crossed": True, "race_completed": False}, {"progress_delta/bonus": .01}),
])
def test_historical_reward_event_predicates(events, expected):
    """Synthetic contexts preserve historical rules even for unusual combinations."""
    from core.scenario import load_yaml_config
    config = load_yaml_config("configs/reward/solo_completion_historical.yaml")
    context = dict(timestep=.05, done=True, terminated=True, info={
        "centerline": {"progress_delta": .01}, "track_limits": {"exceeded": False}, **events})
    reward, breakdown = RewardComposer.from_config(config).compute(context)
    expected = {**expected, "step_time/penalty": -.00025 * .05}
    assert breakdown == pytest.approx(expected)
    assert reward == pytest.approx(sum(expected.values()))


def test_historical_cleanliness_can_be_disabled_and_semantics_are_validated():
    from wrappers.rewards.completion import LapCompletionComponent, ProgressDeltaBonusComponent
    component = LapCompletionComponent(dict(event_semantics="f110_71a3e0f", bonus=2., require_clean=False))
    assert component.compute({"info": dict(lap_crossed=True, race_completed=True, collision=True,
                                           terminal_reason="collision")}) == {"lap_completion/bonus": 2.}
    for cls in (LapCompletionComponent, ProgressDeltaBonusComponent):
        with pytest.raises(ValueError, match="event_semantics"):
            cls({"event_semantics": "historical_typo"})


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


def native_gae(rewards, values, next_values, terminated, truncated):
    from tensordict import TensorDict
    from torchrl.objectives.value import GAE

    n = len(rewards)
    data = TensorDict({
        "state_value": torch.tensor(values).reshape(n, 1),
        "next": TensorDict({
            "state_value": torch.tensor(next_values).reshape(n, 1),
            "reward": torch.tensor(rewards).reshape(n, 1),
            "terminated": torch.tensor(terminated).reshape(n, 1),
            "done": torch.tensor([a or b for a, b in zip(terminated, truncated)]).reshape(n, 1),
        }, [n]),
    }, [n])
    GAE(gamma=0.9, lmbda=1.0, value_network=None)(data)
    return data["advantage"].flatten(), data["value_target"].flatten()


def test_truncation_bootstraps_final_state_without_credit_from_the_next_episode():
    advantages, returns = native_gae(
        rewards=[1., 2.], values=[2., 1000.], next_values=[3., 999.],
        terminated=[False, True], truncated=[True, False],
    )
    torch.testing.assert_close(advantages, torch.tensor([1.7, -998.0]))
    torch.testing.assert_close(returns, torch.tensor([3.7, 2.0]))


def test_collection_cut_bootstraps_without_a_task_boundary():
    advantages, returns = native_gae(
        rewards=[1.], values=[2.], next_values=[3.],
        terminated=[False], truncated=[False],
    )
    torch.testing.assert_close(advantages, torch.tensor([1.7]))
    torch.testing.assert_close(returns, torch.tensor([3.7]))


def test_joint_team_credit_continues_after_an_individual_learner_finishes():
    from agents.torchrl_mappo import TorchRLMAPPOAgent

    agent = TorchRLMAPPOAgent(
        obs_dim=2, global_state_dim=1,
        action_low=np.full(2, -1, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32), agent_ids=LEARNERS,
        params={"n_steps": 4, "hidden_dims": [4], "device": "cpu",
                "critic_mode": "shared_team", "reward_mode": "team_shared",
                "team_return_mode": "joint", "gamma": 0.9, "gae_lambda": 1.0},
    )
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
    from test_torchrl_mappo import transition
    data = torch.stack([transition(agent, ids, index, index + 1, terminal=index == 2)
                        for index, ids in enumerate((LEARNERS, ["car_1"], ["car_1"]))])
    with torch.no_grad():
        agent.gae(data)
    torch.testing.assert_close(data["agents", "value_target"][:, 0, 0], torch.tensor([5.23, 4.7, 3.0]))
    assert data["agents", "active"][:, 0].sum() == 1
    assert data["agents", "active"][:, 1].sum() == 3


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


def test_wheel_acceleration_units_bounds_and_decision_timing():
    composer = ActionComposer.from_config(np.array([-.4, -40.]), np.array([.4, 20.]),
        {"speed_control": "wheel_acceleration", "max_wheel_acceleration": 100.,
         "max_wheel_deceleration": 100., "prevent_reverse": True}, decision_dt=.15)
    # One decision lasts three 0.05 s physics steps; conversion happens once.
    action = np.array([1., 1.], dtype=np.float32)
    for expected in (15., 20., 20.):
        np.testing.assert_allclose(composer.process(action), [.4, expected])
    for expected in (5., 0., 0.):
        assert composer.process(np.array([-1., -1.], dtype=np.float32))[1] == expected
    composer.reset()
    assert composer.process(action)[1] == 15.
    np.testing.assert_array_equal(action, [1., 1.])


def test_motion_lap_tracking_orders_finishers_and_preserves_first_terminal_cause():
    from env.centerline_state import LapTracker, parse_finish_line
    from test_swept_collision import linear_trace
    lifecycle = RaceLifecycle(("first_slot", "second_slot"), target_laps=1)
    finish = parse_finish_line({"start": [0., -1.], "end": [0., 1.],
                               "direction": [1., 0.], "hysteresis": .1, "trigger_speed": .5})
    tracker = LapTracker(lifecycle.agent_ids, finish, lifecycle)
    tracker.reset(np.array([-1., -1.]), np.zeros(2))
    paths = []
    for end in (.5, 2.):
        pose = linear_trace((-1., 0., 0.), (end, 0., 0.))
        state = np.zeros((1, 4, 6))
        state[:, :, :3] = pose
        state[0, 0, 3] = (end + 1.) / .05
        paths.append((pose, state, True, .8))
    crossed = tracker.update(np.array([.2, 1.4]), np.zeros(2), np.zeros(2), np.zeros(2),
                             step=0, motion_paths=paths)
    assert all(crossed.values())
    assert [lifecycle.records[aid].finish_position for aid in lifecycle.agent_ids] == [2, 1]
    for aid in lifecycle.agent_ids:
        assert not lifecycle.record_collision(aid, step=0)
        assert lifecycle.records[aid].terminal_reason.value == "race_complete"
    # A same-fraction reached finish retains historical finish-first precedence.
    lifecycle.reset()
    tracker.reset(np.array([-1., -1.]), np.zeros(2))
    paths[0] = (*paths[0][:3], 2./3.)
    crossed = tracker.update(np.array([0., 1.4]), np.zeros(2), np.zeros(2), np.zeros(2),
                             step=0, motion_paths=paths)
    assert crossed["first_slot"]


def test_motion_lap_tracking_reverse_debt_initial_lap_and_segment_root_deduplication():
    from env.centerline_state import LapTracker, parse_finish_line
    from test_swept_collision import linear_trace
    lifecycle = RaceLifecycle(("car",), target_laps=3)
    finish = parse_finish_line({"start": [0., -1.], "end": [0., 1.], "direction": [1., 0.],
                               "hysteresis": .1})
    tracker = LapTracker(("car",), finish, lifecycle, count_initial_crossing_as_lap=False)
    tracker.reset(np.array([-1.]), np.zeros(1))
    def move(start, end, step):
        pose = np.concatenate([linear_trace((start, 0., 0.), (0., 0., 0.)),
                               linear_trace((0., 0., 0.), (end, 0., 0.))])
        state = np.zeros((2, 4, 6))
        state[:, :, :3] = pose
        state[:, 0, 3] = 1.
        return tracker.update(np.array([end]), np.zeros(1), np.ones(1), np.zeros(1),
                              step=step, motion_paths=[(pose, state, True, 1.)])
    assert not move(-1., 1., 0)["car"]  # first crossing starts, not completes, a lap
    assert lifecycle.records["car"].lap_start_step == 0
    move(1., -1., 1)
    assert tracker._reverse_crossings[0] == 1
    assert not move(-1., 1., 2)["car"]  # only repay one reverse crossing
    assert tracker._reverse_crossings[0] == 0
    assert lifecycle.records["car"].lap_count == 0


def test_frenet_velocity_rotation_is_consistent_with_body_frame_observations():
    from env.centerline_state import CenterlineProgressTracker
    for yaw in (0., .7, -1.2):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        centerline = np.array([[-10., 0., 0.], [0., 0., 0.], [10., 0., 0.]])
        centerline[:, :2] = centerline[:, :2] @ rotation.T
        centerline[:, 2] = yaw
        position = rotation @ [1., .4]
        tracker = CenterlineProgressTracker(["car"])
        facts = tracker.update(centerline, np.array([position[0]]), np.array([position[1]]),
            np.array([yaw + .3]), np.array([2.]), np.array([.5]), {"car": 0})["car"]
        assert facts["vs"] == pytest.approx(2*np.cos(.3) - .5*np.sin(.3), abs=1e-6)
        assert facts["vd"] == pytest.approx(2*np.sin(.3) + .5*np.cos(.3), abs=1e-6)
        assert facts["d"] == pytest.approx(.4, abs=1e-6)
