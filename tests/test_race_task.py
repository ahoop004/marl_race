from types import SimpleNamespace

import numpy as np
import pytest

from env.types import GlobalState, StepFacts
from tasks import RaceTask
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.ego import LidarComponent


class ScriptedEnv:
    possible_agents = ("learner", "opponent")
    trainable_agents = ("learner",)
    fixed_policy_agents = ("opponent",)
    timestep = 0.05

    def __init__(self, events):
        self.events = events
        self.vector = np.zeros(1, dtype=np.float32)
        self.raw = {aid: {"lidar": self.vector, "scans": self.vector} for aid in self.possible_agents}
        self.infos = {aid: {"nested": {"tick": self.vector}} for aid in self.possible_agents}
        self.calls = []

    def reset(self, seed=None, options=None):
        self.reset_args = (seed, options)
        self.vector.fill(0)
        self.tick, self.episode_done = 0, False
        self.agents = list(self.possible_agents)
        self.calls.clear()
        return self.raw, self.infos

    def get_global_state(self):
        return GlobalState(self.possible_agents, self.vector)

    def step(self, actions):
        self.calls.append({aid: action.copy() for aid, action in actions.items()})
        self.tick += 1
        self.vector.fill(self.tick)
        removed, done, term, trunc = self.events.get(self.tick, ((), False, (), ()))
        self.agents = [aid for aid in self.agents if aid not in removed]
        self.episode_done = done
        if done:
            self.agents = []
        terms = {aid: aid in term for aid in self.possible_agents}
        truncs = {aid: aid in trunc for aid in self.possible_agents}
        self.last_step_facts = StepFacts({}, self.get_global_state(), {}, terms, truncs, self.infos)
        return self.raw, {}, terms, truncs, self.infos


class TickReward:
    def reset(self):
        self.contexts = []

    def compute(self, context, team=False):
        if team:
            return 10.0, {"shared": 10.0}
        self.contexts.append(context)
        value = float(context["info"]["nested"]["tick"][0])
        return value, {"tick": value}


def make_task(events, repeat=3, team=False):
    env = ScriptedEnv(events)
    controller = SimpleNamespace(calls=0, resets=0)

    def reset():
        controller.resets += 1

    def act(obs):
        controller.calls += 1
        raise RuntimeError("Exercise the existing zero-action fallback")

    controller.reset, controller.act = reset, act
    action = ActionComposer.from_config(
        np.array([-1, 0], dtype=np.float32), np.array([1, 5], dtype=np.float32),
        {"speed_control": "acceleration", "max_acceleration": 2, "max_deceleration": 2},
        decision_dt=env.timestep * repeat,
    )
    reward = TickReward()
    task = RaceTask(
        env, policy_agents=["learner"], fixed_controllers={"opponent": controller},
        obs_composers={"learner": ObservationComposer([LidarComponent(1, 10, normalize=False)])},
        reward_composers={"learner": reward}, action_composers={"learner": action},
        action_repeat=repeat, team_reward_agent_id="learner" if team else None,
    )
    return task, controller, reward


ACTION = {"learner": np.array([0, 1], dtype=np.float32)}


def test_repeat_accumulates_rewards_holds_commands_and_detaches_snapshots():
    task, controller, reward = make_task({}, team=True)
    initial = task.reset(seed=7, options={"episode_index": 3})
    observed = []
    result = task.step(ACTION, on_physics_step=observed.append)
    decision = result.decisions["learner"]
    assert decision.individual_reward == 6
    assert decision.reward_components == {"tick": 6}
    assert result.team_reward == 30
    assert result.team_reward_components == {"shared": 30}
    assert (result.agent_steps, result.physics_steps, result.after.decision_steps) == (1, 3, 1)
    assert result.elapsed_seconds == pytest.approx(0.15)
    assert observed == list(result.substeps)
    assert controller.calls == controller.resets == 1
    assert task.env.reset_args == (7, {"episode_index": 3})
    assert all(float(context["obs"][0]) == 0 for context in reward.contexts)
    for actions in task.env.calls:
        np.testing.assert_allclose(actions["learner"], [0, 0.3])
        np.testing.assert_array_equal(actions["opponent"], [0, 0])
    task.step(ACTION)
    task.reset()
    for snapshot, tick in ((initial, 0), (result.after, 3)):
        assert snapshot.raw_observations["learner"]["lidar"][0] == tick
        assert snapshot.raw_observations["learner"]["lidar"] is snapshot.raw_observations["learner"]["scans"]
        assert snapshot.infos["learner"]["nested"]["tick"][0] == tick
        assert snapshot.global_state.vector[0] == tick
    assert [substep.facts.info["learner"]["nested"]["tick"][0]
            for substep in result.substeps] == [1, 2, 3]
    assert decision.next_observation[0] == 3
    np.testing.assert_allclose(task.step(ACTION).decisions["learner"].action_physical, [0, 0.3])


@pytest.mark.parametrize("removed", ["learner", "opponent"])
def test_retirement_interrupts_repeat_and_allows_survivor_or_fixed_only_steps(removed):
    task, controller, _ = make_task({1: ((removed,), False, (removed,), ())})
    task.reset()
    result = task.step(ACTION)
    assert result.physics_steps == 1
    assert result.decisions["learner"].individual_reward == 1
    assert result.decisions["learner"].next_observation[0] == 1
    assert not result.after.episode_done
    if removed == "learner":
        assert result.decisions["learner"].terminated
        assert result.after.observations == {}
        continuation = task.step({})
        assert continuation.decisions == {}
        assert continuation.agent_steps == 0
        assert controller.calls == 2
        assert all(set(actions) == {"opponent"} for actions in task.env.calls[1:])
    else:
        continuation = task.step(ACTION)
        assert set(continuation.decisions) == {"learner"}
        assert controller.calls == 1
        assert all(set(actions) == {"learner"} for actions in task.env.calls[1:])


@pytest.mark.parametrize("truncs,boundary", [(('learner',), False), ((), True)])
def test_episode_closure_retains_final_observation_and_physical_flags(truncs, boundary):
    task, _, _ = make_task({1: ((), True, (), truncs)})
    task.reset()
    result = task.step(ACTION)
    decision = result.decisions["learner"]
    assert result.physics_steps == 1
    assert result.after.episode_done and not result.after.agents
    assert decision.truncated and not decision.terminated
    assert decision.next_observation[0] == 1
    assert (decision.info.get("task_boundary") == "episode_policy") is boundary
    assert result.substeps[0].facts.truncations["learner"] is (not boundary)
    with pytest.raises(RuntimeError):
        task.step({})


def test_all_actions_are_validated_before_any_state_changes():
    task, controller, _ = make_task({})
    with pytest.raises(RuntimeError):
        task.step(ACTION)
    task.reset()
    for invalid in ({}, {"opponent": ACTION["learner"]},
                    {"learner": np.array([0, np.nan], dtype=np.float32)},
                    {"learner": np.array([0, 2], dtype=np.float32)},
                    {"learner": np.zeros(2, dtype=np.float64)}):
        with pytest.raises(ValueError):
            task.step(invalid)
    assert controller.calls == 0
    assert task.env.tick == 0
    np.testing.assert_allclose(task.step(ACTION).decisions["learner"].action_physical, [0, 0.3])
