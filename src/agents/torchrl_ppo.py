from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import IndependentNormal, ProbabilisticActor
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from agents.common.outputs import PolicyOutput
from agents.common.ppo_policy import PPOPolicy
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


class TorchRLPPOAgent(PPOPolicy):
    def __init__(self, obs_dim, action_low, action_high, params):
        super().__init__(obs_dim, action_low, action_high, params, training=True)
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
        # Native environments keep the singleton learner in an agents group.
        self.collection_policy = TensorDictSequential(
            TensorDictModule(lambda obs: obs.squeeze(-2),
                             in_keys=[("agents", "observation")], out_keys=["observation"]),
            self.policy,
            TensorDictModule(lambda action: action.unsqueeze(-2),
                             in_keys=["action"], out_keys=[("agents", "action")]),
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
        self._pending_steps = 0

    def update(self, rollout: TensorDictBase):
        return self.update_rollouts([rollout])

    @torch.no_grad()
    def sample_batch(self, observations, deterministic=False):
        observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        data = TensorDict({"observation": observation}, [len(observation)])
        with set_exploration_type(ExplorationType.MODE if deterministic else ExplorationType.RANDOM):
            self.policy(data)
        outputs = torch.cat((data["action"], data["raw_action"], data["raw_log_prob"][:, None],
                             self.critic(observation)[:, None]), dim=1).cpu().numpy()
        return PolicyOutput(outputs[:, :self.action_dim], outputs[:, -2],
                            outputs[:, self.action_dim:2 * self.action_dim].copy(), outputs[:, -1])

    def update_rollouts(self, rollouts):
        # GAE must see each race's time axis independently, before flattening.
        for rollout in rollouts:
            self._append_rollout(rollout)
        if self._pending_steps < self.min_rollout_steps:
            return {}
        return self.flush_pending_update()

    def _append_rollout(self, rollout):
        if not isinstance(rollout, TensorDictBase):
            raise TypeError("TorchRL PPO requires a TensorDict rollout")
        data = rollout.to(self.device).clone()
        if ("agents", "observation") in data.keys(True):
            data["observation"] = data["agents", "observation"].squeeze(-2)
            data["next", "observation"] = data["next", "agents", "observation"].squeeze(-2)
            data["next", "reward"] = data["next", "agents", "reward"].squeeze(-2)
        with torch.no_grad():
            self.gae(data)
        self._pending_batches.append(data.reshape(-1))
        self._pending_steps += data.numel()

    def flush_pending_update(self):
        if not self._pending_batches:
            return {}
        data = torch.cat(self._pending_batches, dim=0)
        self._pending_batches.clear()
        self._pending_steps = 0
        return optimize_ppo(self, data)
