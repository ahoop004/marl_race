import math

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")
pytest.importorskip("torchrl")
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from agents.ppo import PPOAgent
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
    legacy = PPOAgent(1, learner.action_low, learner.action_high, {"hidden_dims": [], "device": "cpu"})
    legacy.load(str(checkpoint))
    observation = np.array([0.3], dtype=np.float32)
    np.testing.assert_array_equal(legacy.predict(observation), learner.predict(observation))
    legacy.save(str(checkpoint))
    restored = agent(vf_coef=vf_coef)
    restored.load(str(checkpoint))
    assert restored.optimizer.state_dict()["state"]
    np.testing.assert_array_equal(restored.predict(observation), learner.predict(observation))


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
        task.env, "learner", agent(**params), task.fixed_controllers,
        task.obs_composers["learner"], task.reward_composers["learner"],
        task.action_composers["learner"], hooks=[hook],
    )
    return result, controller


@pytest.mark.parametrize("transitions", [False, True])
def test_native_trainer_exact_budget_resets_and_hook_boundaries(transitions):
    hook = Capture(transitions)
    training, controller = trainer(hook)
    training.train(total_steps=7)
    assert training.collected_steps == controller.calls == 7
    assert controller.resets == len(hook.starts) == 3
    assert [entry[1] for entry in hook.episodes] == [6, 6]
    assert [entry[3]["episode_steps"] for entry in hook.episodes] == [3, 3]
    assert [update["train/rollout_steps"] for update in hook.updates] == [4, 3]
    assert [update["train/environment_steps"] for update in hook.updates] == [4, 7]
    assert hook.ended
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
