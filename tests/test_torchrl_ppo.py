import math

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")
pytest.importorskip("torchrl")
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from agents.common.ppo_policy import PPOPolicy
from agents.torchrl_ppo import TorchRLPPOAgent
from training.hooks import TrainingHook
from training.torchrl_ppo_trainer import TorchRLPPOTrainer
from test_race_task import make_task


def agent(**params):
    return TorchRLPPOAgent(1, np.array([-1, 0]), np.array([1, 5]), {
        "hidden_dims": [], "n_steps": 4, "n_epochs": 2, "batch_size": 2,
        "device": "cpu", **params,
    })


def rollout(policy, observations, *, done=None, terminated=None):
    n = len(observations)
    data = TensorDict({"observation": torch.tensor(observations, dtype=torch.float32).reshape(n, 1)}, [n])
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        policy(data)
    data["next"] = TensorDict({
        "observation": data["observation"] + 1,
        "reward": torch.ones(n, 1),
        "done": torch.tensor(done or [False] * n).reshape(n, 1),
        "terminated": torch.tensor(terminated or [False] * n).reshape(n, 1),
    }, [n])
    return data


def test_native_gae_bootstraps_truncations_and_budget_cuts_without_crossing_resets():
    learner = agent(gamma=0.9, gae_lambda=0.8)
    with torch.no_grad():
        learner.critic.net[0].weight.fill_(1)
        learner.critic.net[0].bias.zero_()
    data = rollout(learner.policy, [2, 3, 100, 5],
                   done=[False, True, True, False], terminated=[False, False, True, False])
    data["next", "observation"] = torch.tensor([[3.], [4.], [999.], [7.]])
    data["next", "reward"] = torch.tensor([[1.], [2.], [3.], [4.]])
    with torch.no_grad():
        learner.gae(data)
    torch.testing.assert_close(data["advantage"].squeeze(-1), torch.tensor([3.572, 2.6, -97., 5.3]))


def test_native_clipped_loss_matches_existing_objective_for_saturated_actions():
    learner = agent()
    with torch.no_grad():
        learner.actor.net[0].weight.zero_()
        learner.actor.net[0].bias.fill_(20)
    data = rollout(learner.policy, [0, 1, 2])
    assert torch.all(data["action"] == 1)
    old_log_prob, _ = learner.actor.evaluate_actions(data["observation"], data["action"], data["raw_action"])
    old_log_prob = old_log_prob.detach()
    data["advantage"] = torch.tensor([[2.], [-1.], [3.]])
    data["value_target"] = torch.tensor([[1.], [2.], [3.]])
    with torch.no_grad():
        learner.actor.net[0].bias.add_(0.2)
    log_prob, _ = learner.actor.evaluate_actions(data["observation"], data["action"], data["raw_action"])
    ratio = (log_prob - old_log_prob).exp()
    advantage = data["advantage"].squeeze(-1)
    expected = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2)).mean()
    losses = learner.loss_module(data)
    torch.testing.assert_close(losses["loss_objective"], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(losses["loss_critic"], learner.vf_coef * torch.nn.functional.mse_loss(
        learner.critic(data["observation"]), data["value_target"].squeeze(-1)))


@pytest.mark.parametrize("vf_coef", [0, 0.5])
def test_native_update_handles_partial_minibatches_and_legacy_checkpoints(vf_coef, tmp_path):
    learner = agent(vf_coef=vf_coef)
    data = rollout(learner.policy, [0, 1, 2])
    metrics = learner.update(data)
    assert metrics["train/optimizer_steps"] == 4
    assert all(math.isfinite(value) for value in metrics.values())
    checkpoint = tmp_path / "model.pt"
    learner.save(str(checkpoint))
    evaluation_policy = PPOPolicy(1, learner.action_low, learner.action_high, {"hidden_dims": [], "device": "cpu"})
    evaluation_policy.load(str(checkpoint))
    observation = np.array([0.3], dtype=np.float32)
    np.testing.assert_array_equal(evaluation_policy.predict(observation), learner.predict(observation))
    evaluation_policy.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    assert payload.pop("network")["architecture"] == "mlp"
    torch.save(payload, checkpoint)
    restored = agent(vf_coef=vf_coef)
    restored.load(str(checkpoint))
    assert restored.optimizer.state_dict()["state"]
    np.testing.assert_array_equal(restored.predict(observation), learner.predict(observation))


def test_prediction_uses_actor_distribution_parameters(monkeypatch):
    learner = agent()
    mean = torch.tensor([[0.25, -0.75]])
    monkeypatch.setattr(learner.actor, "forward", lambda observations: (mean, torch.ones(2)))
    np.testing.assert_array_equal(learner.predict(np.zeros(1, dtype=np.float32)), mean.tanh()[0].numpy())


class Capture(TrainingHook):
    def __init__(self, transitions):
        self.requires_transition_record = transitions
        self.starts, self.steps, self.episodes, self.updates = [], [], [], []
        self.ended = False

    def on_episode_start(self, metadata):
        self.starts.append(metadata)

    def on_step(self, record):
        self.steps.append(record)

    def on_episode_end(self, episode, reward, info, metrics):
        self.episodes.append((episode, reward, info, metrics))

    def on_update(self, metrics):
        self.updates.append(dict(metrics))

    def on_training_end(self):
        self.ended = True


def trainer(hook, **params):
    task, controller, _ = make_task({3: ((), True, (), ("learner", "opponent"))}, repeat=1)
    task.env.render_mode = None
    task.env.close = lambda: None
    result = TorchRLPPOTrainer(
        task, agent(**params), hooks=[hook],
    )
    return result, controller


@pytest.mark.parametrize("transitions", [False, True])
def test_native_trainer_exact_budget_resets_and_hook_boundaries(transitions, monkeypatch):
    hook = Capture(transitions)
    training, controller = trainer(hook)
    fragments = []
    update = training.agent.update

    def capture(data):
        fragments.append(data.clone())
        return update(data)

    monkeypatch.setattr(training.agent, "update", capture)
    training.train(total_steps=7)
    assert training.collected_steps == controller.calls == 7
    assert controller.resets == len(hook.starts) == 3
    assert [entry[1] for entry in hook.episodes] == [6, 6]
    assert [entry[3]["episode_steps"] for entry in hook.episodes] == [3, 3]
    assert [update["train/rollout_steps"] for update in hook.updates] == [4, 3]
    assert [update["train/environment_steps"] for update in hook.updates] == [4, 7]
    assert hook.ended
    data = torch.cat(fragments)
    assert data["observation"].flatten().tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert data["next", "observation"].flatten().tolist() == [1, 2, 3, 1, 2, 3, 1]
    assert data["next", "done"].flatten().tolist() == [False, False, True, False, False, True, False]
    assert not data["next", "terminated"].any()
    if transitions:
        assert len(hook.steps) == 7
        assert [record.step_idx for record in hook.steps] == [0, 1, 2, 0, 1, 2, 0]
        assert [record.next_obs[0] for record in hook.steps] == [1, 2, 3, 1, 2, 3, 1]
        assert hook.steps[2].truncated and not hook.steps[2].terminated
        assert not hook.steps[-1].truncated


def test_evaluation_gate_stops_collection_and_preserves_evaluated_weights():
    class Gate(Capture):
        def on_update(self, metrics):
            super().on_update(metrics)
            self.should_stop = True
            self.weights = {key: value.clone() for key, value in training.agent.actor.state_dict().items()}

    hook = Gate(False)
    training, _ = trainer(hook)
    training.train(total_steps=7)
    assert training.collected_steps == 4
    assert len(hook.updates) == 1
    assert hook.ended
    for key, value in training.agent.actor.state_dict().items():
        torch.testing.assert_close(value, hook.weights[key], rtol=0, atol=0)


def test_episode_budget_flushes_a_deferred_update_with_final_learning_rate():
    hook = Capture(False)
    training, controller = trainer(hook, min_rollout_steps=10, lr_schedule="linear",
                                   learning_rate=1e-3, learning_rate_end=1e-4)
    training.train(n_episodes=2)
    assert training.collected_steps == 6
    assert controller.resets == len(hook.episodes) == 2
    assert len(hook.updates) == 1
    assert hook.updates[0]["train/rollout_steps"] == 6
    assert hook.updates[0]["train/updates"] == 1
    assert hook.updates[0]["train/learning_rate"] == pytest.approx(1e-4)


def test_rollout_fragments_bootstrap_independently_with_native_log_probabilities(monkeypatch):
    learner = agent(gamma=0.9, gae_lambda=1)
    with torch.no_grad():
        learner.critic.net[0].weight.fill_(1)
        learner.critic.net[0].bias.zero_()
    fragments = []
    for observations, next_observations, rewards in (([2, 3], [3, 4], [1, 2]), ([100], [7], [4])):
        data = rollout(learner.policy, observations)
        data["next", "observation"] = torch.tensor(next_observations, dtype=torch.float32).reshape(-1, 1)
        data["next", "reward"] = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        fragments.append(data)
    captured = []
    monkeypatch.setattr("agents.torchrl_ppo.optimize_ppo", lambda agent, data: captured.append(data.clone()) or {})
    learner.update_rollouts(fragments)
    torch.testing.assert_close(captured[0]["value_target"].flatten(), torch.tensor([6.04, 5.6, 10.3]))
    log_probs = learner.probabilistic_actor.get_dist(captured[0]).log_prob(captured[0]["raw_action"])
    torch.testing.assert_close(log_probs, captured[0]["raw_log_prob"])


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_serial_trainers_stop_after_update_and_keep_evaluated_weights(algorithm):
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task

    scenarios = Path(__file__).resolve().parents[1] / "scenarios"
    filename = "ppo_lap_completion_pretrain.yaml" if algorithm == "ppo" else "mappo_2v2_completion_scratch.yaml"
    overrides = [f"algorithm.backend=torchrl",
                 "experiment.total_steps=7", "environment.max_steps=10", "evaluation.enabled=false",
                 "++agents.car_0.params.n_steps=4"]
    if algorithm == "mappo":
        pytest.importorskip("torchrl.objectives.multiagent")
        overrides += ["++agents.car_1.params.n_steps=4", "controllers.racing_mpc.max_evaluations=10"]
    scenario = load_and_expand_scenario(str(scenarios / filename), overrides=overrides)
    task = create_race_task(scenario, scenario_dir=scenarios)
    ids = list(task.possible_agents)
    params = {"device": "cpu", "hidden_dims": [8], "n_steps": 4, "n_epochs": 1, "batch_size": 4}
    obs_dim = task.obs_composers[ids[0]].obs_dim
    bounds = task.env.action_spaces[ids[0]]
    hook = Capture(False)

    def stop(metrics):
        hook.updates.append(dict(metrics))
        hook.should_stop = True
        hook.weights = {key: value.clone() for key, value in learner.actor.state_dict().items()}

    hook.on_update = stop
    try:
        if algorithm == "ppo":
            learner = TorchRLPPOAgent(obs_dim, bounds.low, bounds.high, params)
            training = TorchRLPPOTrainer(task, learner, hooks=[hook])
        else:
            from agents.torchrl_mappo import TorchRLMAPPOAgent
            from training.torchrl_mappo_trainer import TorchRLMAPPOTrainer

            state = task.env.get_global_state()
            params.update(critic_mode="shared_team", reward_mode="team_shared", team_return_mode="joint",
                          _global_state_contract_version=state.metadata["vector_contract_version"],
                          _observation_dims={aid: task.obs_composers[aid].obs_dim for aid in ids},
                          _observation_contracts={aid: task.obs_composers[aid].contract for aid in ids})
            learner = TorchRLMAPPOAgent(obs_dim, len(state.vector), bounds.low, bounds.high, ids, params)
            training = TorchRLMAPPOTrainer(task, learner, hooks=[hook])
        training.train(total_steps=7)
        assert len(hook.updates) == 1 and hook.updates[0]["train/environment_steps"] == 4
        assert hook.ended and not hook.episodes
        for key, value in learner.actor.state_dict().items():
            torch.testing.assert_close(value, hook.weights[key], rtol=0, atol=0)
    finally:
        task.close()
