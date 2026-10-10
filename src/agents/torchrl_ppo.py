from __future__ import annotations

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import IndependentNormal, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from agents.ppo import PPOAgent
from agents.torchrl_updates import optimize_ppo


class _PolicyParameters(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation):
        mean, scale = self.actor(observation)
        return mean, scale.expand_as(mean)


class _Value(torch.nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, observation):
        return self.critic(observation).unsqueeze(-1)


class TorchRLPPOAgent(PPOAgent):
    def _make_buffer(self):
        # Collection and storage use TensorDicts instead of the legacy buffer.
        return None

    def __init__(self, obs_dim, action_low, action_high, params):
        super().__init__(obs_dim, action_low, action_high, params)
        if min(self.n_steps, self.n_epochs, self.batch_size) < 1:
            raise ValueError("PPO rollout size, epochs, and batch size must be positive")
        parameters = TensorDictModule(
            _PolicyParameters(self.actor), in_keys=["observation"], out_keys=["loc", "scale"],
        )
        self.probabilistic_actor = ProbabilisticActor(
            parameters, in_keys=["loc", "scale"], out_keys=["raw_action"],
            distribution_class=IndependentNormal,
            return_log_prob=True, log_prob_key="raw_log_prob",
        )
        self.policy = TensorDictSequential(
            self.probabilistic_actor,
            TensorDictModule(torch.tanh, in_keys=["raw_action"], out_keys=["action"]),
        )
        self.value_module = TensorDictModule(
            _Value(self.critic), in_keys=["observation"], out_keys=["state_value"],
        )
        # Tanh's Jacobian cancels in PPO ratios. Stored latent samples avoid
        # reconstructing saturated actions; the existing bounded entropy stays.
        self.loss_module = ClipPPOLoss(
            self.probabilistic_actor, self.value_module, clip_epsilon=self.clip_range,
            entropy_bonus=False, critic_coeff=self.vf_coef, loss_critic_type="l2",
            normalize_advantage=False, functional=False,
        )
        self.loss_module.set_keys(action="raw_action", sample_log_prob="raw_log_prob")
        self.gae = GAE(
            gamma=self.gamma, lmbda=self.gae_lambda, value_network=self.value_module,
            average_gae=False, auto_reset_env=False,
        )
        self._pending_batches = []

    def update(self, rollout: TensorDictBase):
        if not isinstance(rollout, TensorDictBase):
            raise TypeError("TorchRL PPO requires a TensorDict rollout")
        data = rollout.to(self.device).clone()
        with torch.no_grad():
            self.gae(data)
        self._pending_batches.append(data.reshape(-1))
        self._pending_steps += data.numel()
        if self._pending_steps < self.min_rollout_steps:
            return {}
        return self.flush_pending_update()

    def flush_pending_update(self):
        if not self._pending_batches:
            return {}
        data = torch.cat(self._pending_batches, dim=0)
        self._pending_batches.clear()
        self._pending_steps = 0
        return optimize_ppo(self, data)
