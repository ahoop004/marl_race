"""Reward parity, episode completion and inference isolation across callers."""
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters.rewards import RewardMapping
from adapters.native_torchrl import NativeRaceTorchRLEnv
from torchrl.envs import step_mdp
from tasks import RaceTask
from training.evaluation import run_evaluation_episode
from training.mappo_evaluator import DeterministicMAPPOEvaluator
from training.ppo_evaluator import DeterministicPPOEvaluator
from test_race_task import ScriptedEnv, TickReward, make_task
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.ego import LidarComponent


ROOT = Path(__file__).resolve().parents[1]


def two_policy_task():
    core = ScriptedEnv({1: (("learner",), False, ("learner",), ()),
                        2: ((), True, (), ("opponent",))})
    core.render_mode, core.map_name, core.max_steps = None, "scripted", 2
    core.close = lambda: None
    ids = core.possible_agents
    return RaceTask(core, policy_agents=ids, fixed_controllers={},
        obs_composers={aid: ObservationComposer([LidarComponent(1, 10, normalize=False)]) for aid in ids},
        reward_composers={aid: TickReward() for aid in ids},
        action_composers={aid: ActionComposer([]) for aid in ids},
        action_repeat=3, team_reward_agent_id="learner")


@pytest.mark.parametrize("mode,reduction,expected", [
    ("individual", "mean", (1, 2)),
    ("team_shared", "mean", (11, 11)),
    ("team_shared", "sum", (12, 12)),
])
def test_native_torchrl_and_evaluation_deliver_identical_rewards_after_retirement(mode, reduction, expected):
    task = two_policy_task()
    env = NativeRaceTorchRLEnv(task, reward_mode=mode, team_reward_reduction=reduction)
    delivered, trace = {}, []
    try:
        current = env.reset(seed=17)
        while not env.snapshot.episode_done:
            current["agents", "action"] = torch.zeros(len(env.agent_ids), 2)
            transition = env.step(current)
            nxt = transition["next"]
            trace.extend(env.last_step.substeps)
            for aid in env.last_step.decisions:
                row = env.agent_ids.index(aid)
                reward = nxt["agents", "reward"][row].item()
                delivered[aid] = delivered.get(aid, 0.0) + reward
                assert nxt["agents", "individual_reward"][row].item() == task.env.tick
                assert nxt["shared_bonus"].item() == 10
            current = step_mdp(transition)
    finally:
        env.close()
    assert delivered == {"learner": expected[0], "opponent": sum(expected)}
    evaluated = run_evaluation_episode(two_policy_task(),
        lambda ids, obs: {aid: np.zeros(2, dtype=np.float32) for aid in ids},
        episode=4, seed=17, reward_mapping=RewardMapping(mode, reduction))
    assert {aid: facts.reward_total for aid, facts in evaluated.facts.agents.items()} == delivered
    assert evaluated.team_components == {"shared": 20}
    assert evaluated.snapshot.physics_steps == len(trace) == 2
    assert evaluated.snapshot.episode_done
    assert evaluated.facts.agents["opponent"].individual_reward_total == 3
    assert evaluated.facts.agents["opponent"].reward_components == {"tick": 3}
    for actual, reference in zip(evaluated.snapshot.raw_observations.values(), task.env.raw.values()):
        np.testing.assert_array_equal(actual["lidar"], reference["lidar"])


def fixed_race_task():
    task, controller, reward = make_task({
        1: (("learner",), False, ("learner",), ()),
        2: ((), True, (), ("opponent",)),
    })
    task.env.render_mode, task.env.map_name, task.env.max_steps = None, "scripted", 2
    return task, controller, reward


@pytest.mark.parametrize("completion,steps,calls", [("policy", 1, 1), ("race", 2, 2)])
def test_evaluation_completion_preserves_fixed_only_continuation(completion, steps, calls):
    task, controller, reward = fixed_race_task()
    seen = []

    def actions(ids, observations):
        seen.append((ids, deepcopy(observations)))
        return {aid: np.array([0, 1], dtype=np.float32) for aid in ids}

    result = run_evaluation_episode(task, actions, episode=7, seed=23, completion=completion)
    assert result.snapshot.physics_steps == steps
    assert result.facts.steps == steps
    assert controller.calls == calls and controller.resets == 1
    assert len(seen) == 1  # Fixed-only steps produce no policy inference.
    assert result.facts.agents["learner"].reward_total == 1
    assert len(reward.contexts) == 1
    assert task.env.reset_args == (23, {"map_episode_index": 7, "spawn_episode_index": 7})
    assert result.snapshot.episode_done is (completion == "race")
    assert reward.contexts[0]["obs"][0] == 0


@pytest.mark.parametrize("evaluator_class,steps", [(DeterministicPPOEvaluator, 1),
                                                   (DeterministicMAPPOEvaluator, 2)])
def test_checkpoint_evaluation_uses_task_runner_and_restores_inference_state(evaluator_class, steps):
    task, controller, _ = fixed_race_task()
    actor = torch.nn.Linear(1, 2)
    calls = []

    def actions(ids, obs):
        assert not actor.training and not torch.is_grad_enabled()
        calls.append((random.random(), np.random.rand(), torch.rand(1).item()))
        return {aid: np.zeros(2, dtype=np.float32) for aid in ids}

    policy = SimpleNamespace(actor=actor, evaluation_actions=actions)
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    before = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    evaluator = evaluator_class(task=task, episodes=2, base_seed=31).bind_agent(policy)
    first = evaluator.evaluate()
    after = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    assert actor.training
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
    assert task.env.tick == steps and controller.calls == 2 * steps
    second = evaluator.evaluate()
    assert first == second and calls[:2] == calls[2:]
    assert "mean_episode_reward" not in first
    assert "reward" not in first["episode_results"][0]["agents"]["learner"]


def test_checkpoint_evaluation_restores_rng_and_actor_mode_after_failure():
    task, _, _ = fixed_race_task()
    actor = torch.nn.Linear(1, 2)
    actor.eval()
    state = torch.random.get_rng_state().clone()

    def fail(ids, obs):
        torch.rand(1)
        raise RuntimeError("policy failed")

    evaluator = DeterministicPPOEvaluator(task=task, episodes=1, base_seed=0).bind_agent(
        SimpleNamespace(actor=actor, evaluation_actions=fail))
    with pytest.raises(RuntimeError, match="policy failed"):
        evaluator.evaluate()
    assert not actor.training
    torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)


def test_task_creation_does_not_depend_on_learner_algorithm_or_global_rng(monkeypatch):
    from core.agent_roles import AgentRoles
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task

    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    config = load_and_expand_scenario(str(ROOT / "scenarios/ppo_lap_completion_pretrain.yaml"))
    config["agents"]["car_0"]["algorithm"] = "future_learner"
    state = torch.random.get_rng_state().clone()
    task = create_race_task(config, scenario_dir=ROOT / "scenarios",
                            roles=AgentRoles(("car_0",), ()))
    try:
        assert task.possible_agents == ("car_0",)
        assert task._snapshot is None
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
    finally:
        task.close()


def test_task_assigns_one_owner_for_configured_shared_components():
    task = two_policy_task()
    for composer in task.reward_composers.values():
        composer.team_contract = ["shared_tick"]
    rebuilt = RaceTask(task.env, policy_agents=task.possible_agents, fixed_controllers={},
        obs_composers=task.obs_composers, reward_composers=task.reward_composers,
        action_composers=task.action_composers)
    assert rebuilt.team_reward_agent_id == "learner"
    rebuilt.reset()
    result = rebuilt.step({aid: np.zeros(2, dtype=np.float32) for aid in rebuilt.agents})
    assert result.team_reward == 10
    assert result.team_reward_components == {"shared": 10}


def test_selection_evaluator_repeats_fixed_seed_races_and_keeps_training_task_isolated(monkeypatch):
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task
    from training.algorithms import learner_params, create_learner
    from training.selection import create_selection_evaluator

    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    config = load_and_expand_scenario(str(ROOT / "scenarios/mappo_2v2_completion_scratch.yaml"))
    config["environment"]["max_steps"] = 2
    config["evaluation"].update(max_steps=2, episodes=3)
    config["evaluation"].pop("protocols", None)
    config["training_defaults"].update(hidden_dims=[4], device="cpu")
    training_task = create_race_task(config, scenario_dir=ROOT / "scenarios")
    evaluator = None
    try:
        spec = training_task.spec
        params = learner_params(config, spec, "mappo")
        policy = create_learner("mappo", spec, params, training=False)
        evaluator = create_selection_evaluator("mappo", config, ROOT / "scenarios", spec, params).bind_agent(policy)
        expected = evaluator.evaluate()
        assert expected == evaluator.evaluate()
        assert len(expected["episode_results"]) == 3
        assert expected["evaluation_protocol"]["seeds"] == [10042, 10043, 10044]
        assert policy.actor.training
        assert training_task._snapshot is None
    finally:
        if evaluator is not None:
            evaluator.close()
        training_task.close()
