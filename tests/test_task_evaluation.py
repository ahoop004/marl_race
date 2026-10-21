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


@pytest.mark.parametrize("adaptation", ["ppo", "scratch", "full_finetune", "lora"])
def test_solo_actor_extraction_preserves_identity_and_native_deterministic_actions(tmp_path, adaptation, monkeypatch):
    from agents.common.checkpoints import read_checkpoint, solo_evaluation_actor
    from application.evaluate import solo_scenario
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task
    from training.algorithms import learner_params, create_learner

    ppo_config = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml",
        overrides=["network=mlp_base", "environment.lidar_beams=108"])
    with_task = create_race_task(ppo_config)
    try:
        params = learner_params(ppo_config, with_task.spec, "ppo")
        params["device"] = "cpu"
        ppo = create_learner("ppo", with_task.spec, params, training=False)
        source = tmp_path / "ppo.pt"
        ppo.save_checkpoint(source, configuration=ppo_config)
    finally:
        with_task.close()
    if adaptation == "ppo":
        config, policy, checkpoint = ppo_config, ppo, source
        ids = ["car_0"]
    else:
        config = load_and_expand_scenario("scenarios/mappo_2v2_completion_scratch.yaml",
            overrides=["network=mlp_base", "environment.lidar_beams=108", f"adaptation={adaptation}"])
        task = create_race_task(config)
        try:
            params = learner_params(config, task.spec, "mappo")
            params.update(device="cpu", compile_networks=False)
            policy = create_learner("mappo", task.spec, params, training=False)
            if adaptation != "scratch":
                policy.initialize_from_checkpoint(source, scope="actor_only", observation_extension="frenet_neighbors")
            with torch.no_grad():
                if adaptation == "lora":
                    for adapter in policy.actor.adapters[1].values():
                        adapter.B.fill_(0.07)
                else:
                    policy.actor.actors["car_1"].net[-1].bias.add_(0.4)
            checkpoint = tmp_path / "mappo.pt"
            policy.save_checkpoint(checkpoint, configuration=config)
        finally:
            task.close()

        ids = ["car_0", "car_1"]
    payload = read_checkpoint(checkpoint)
    for aid in ids:
        solo = solo_scenario(config, aid)
        task = create_race_task(solo, mode="eval")
        try:
            assert task.physical_agents == (aid,) and task.env.n_agents == 1
            params = learner_params(solo, task.spec, payload["metadata"]["algorithm"])
            params.update(device="cpu", compile_networks=False)
            def forbidden(*args, **kwargs):
                pytest.fail("Actor-only inference attempted to construct a centralized critic")
            monkeypatch.setattr("agents.common.networks.build_critic", forbidden)
            selected = solo_evaluation_actor(payload, task.spec, params, aid)
            snapshot = task.reset(seed=30042)
            obs = snapshot.observations[aid]
            if adaptation != "ppo":
                # Three empty 10-value slots plus the original four-value ego ID.
                np.testing.assert_array_equal(obs[-34:-4], np.zeros(30))
                np.testing.assert_array_equal(obs[-4:], np.eye(4)[int(aid[-1])])
            expected = policy.evaluation_actions([aid], {aid: obs})[aid]
            actual = selected.evaluation_actions([aid], {aid: obs})[aid]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            assert not hasattr(selected, "critic")
            assert selected.identity["adapter_active"] == (adaptation == "lora")
            assert selected.identity["source_adapter_index"] == (int(aid[-1]) if adaptation == "lora" else None)
            bad = deepcopy(payload)
            bad["metadata"]["contracts"]["observations"][aid]["dim"] += 1
            with pytest.raises(ValueError, match="Incompatible"):
                solo_evaluation_actor(bad, task.spec, params, aid)
        finally:
            task.close()

    # Exercise report writing through the existing standalone evaluator with a
    # two-step cap; this checks inference/report plumbing, not policy skill.
    import json
    import application.evaluate as evaluator
    from loggers.console import ConsoleLogger
    native_episode = evaluator.run_evaluation_episode
    def short_episode(task, *args, **kwargs):
        maximum = task.env.max_steps
        task.env.max_steps = 2
        try:
            return native_episode(task, *args, **kwargs)
        finally:
            task.env.max_steps = maximum
    monkeypatch.setattr(evaluator, "run_evaluation_episode", short_episode)
    evaluation_config = deepcopy(config)
    evaluation_config["training_defaults"]["device"] = "cpu"
    folder = tmp_path / "solo_report"
    evaluator.run_evaluation(evaluation_config, SimpleNamespace(checkpoint=str(checkpoint), eval_actor=ids[-1],
        eval_protocol="final", eval_episodes=1, scenario=config["_configuration"]["source"], run_id="solo-test", output_dir=str(folder)),
        ConsoleLogger(verbose=False), ROOT / "scenarios")
    report = json.loads((folder / "evaluation_report.json").read_text())
    assert report["checkpoint_sha256"] == payload["_identity"]["sha256"]
    assert report["actor_identity"]["source_actor"] == ids[-1]
    assert report["target_laps"] == 3 and report["max_steps"] == 16000
    assert report["seeds"] == [30042]
    assert report["episode_results"][0]["race_record"]["physics_steps"] == 2
    assert "stationary" in report["episode_results"][0]["race_record"]["agents"][ids[-1]]


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
