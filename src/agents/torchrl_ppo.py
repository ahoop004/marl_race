from __future__ import annotations

from contextlib import nullcontext

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import IndependentNormal, ProbabilisticActor
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.value import GAE

from agents.common.outputs import PolicyOutput
from agents.common.ppo_policy import PPOPolicy
from agents.common.optimization import make_objective
from agents.torchrl_rollouts import rollout_groups, transfer_rollout_group, batch_gae
from agents.torchrl_updates import (PPO_OPTIMIZATION_KEYS, PPO_OPTIONAL_OPTIMIZATION_KEYS,
                                   optimize_ppo)


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
    # GAE's value-network inputs, reward/boundaries, optional cached values and
    # step discounts; active is used only after GAE. Reporting owns other keys.
    _ROLLOUT_KEYS = (
        "observation", "action", "raw_action", "raw_log_prob", ("agents", "active"),
        ("next", "observation"), ("next", "reward"), ("next", "done"),
        ("next", "terminated"), ("next", "truncated"), "state_value",
        ("next", "state_value"), "steps_to_next_obs", "is_init", ("next", "is_init"),
    )

    def select_rollout(self, rollout, *, retain_episode=False):
        """Select tensor references; the caller still owns the cloning boundary."""
        keys = self._ROLLOUT_KEYS + (("environment_episode",) if retain_episode else ())
        # Selecting an absent nested mask with strict=False would retain an
        # empty collector TensorDict when only traj_ids was present.
        keys += tuple(key for key in PPO_OPTIONAL_OPTIMIZATION_KEYS
                      if rollout.get(key, None) is not None)
        data = rollout.select(*keys, strict=False)
        observation = rollout.get(("agents", "observation"), None)
        if observation is not None:
            data["observation"] = observation.squeeze(-2)
            data["next", "observation"] = rollout["next", "agents", "observation"].squeeze(-2)
            for name in ("reward", "done", "terminated", "truncated"):
                value = rollout.get(("next", "agents", name), None)
                if value is not None:
                    # Physical race closure can follow the learner's retirement.
                    data["next", name] = value.squeeze(-2)
        return data

    def __init__(self, obs_dim, action_low, action_high, params):
        super().__init__(obs_dim, action_low, action_high, params, training=True)
        self.frames_per_batch = int(params.get("frames_per_batch", 256))
        if min(self.n_steps, self.n_epochs, self.batch_size, self.frames_per_batch) < 1:
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
        self.loss_module = make_objective(
            self.algorithm, self.loss, self.critic_loss, self.probabilistic_actor, self.value_module,
            clip_epsilon=self.clip_range, entropy_bonus=False, critic_coeff=self.vf_coef,
            normalize_advantage=False, functional=False,
        )
        self.loss_module.set_keys(action="raw_action", sample_log_prob="raw_log_prob")
        self.gae = GAE(
            gamma=self.gamma, lmbda=self.gae_lambda, value_network=self.value_module,
            average_gae=False, auto_reset_env=False,
            # Eager vmap over a compiled forward is unsupported by PyTorch.
            # TorchRL's equivalent sequential value calls preserve GAE maths.
            deactivate_vmap=self.compilation_config["critic"],
        )
        self._pending_batches = []
        self._pending_steps = 0
        self._pending_environment_steps = 0

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
        self._append_rollouts(rollouts)
        if self._pending_steps < self.min_rollout_steps and not getattr(self, "distributed", None):
            return {}
        return self.flush_pending_update()

    def _append_rollout(self, rollout):
        self._append_rollouts([rollout])

    def _append_rollouts(self, rollouts):
        region = getattr(self, "profile_region", None)
        with region("rollout_preparation") if region else nullcontext():
            for group in rollout_groups(self, rollouts):
                data, all_active = transfer_rollout_group(self, group)
                environment_steps = data.numel()
                batch_gae(self, data, len(group))
                with region("post_gae_flatten_mask") if region else nullcontext():
                    active = data.get(("agents", "active"), None)
                    keys = PPO_OPTIMIZATION_KEYS + tuple(key for key in PPO_OPTIONAL_OPTIMIZATION_KEYS
                                                         if data.get(key, None) is not None)
                    # Trajectory-major flattening is exactly the original
                    # concatenation order. Compute dynamic indices only once.
                    data = data.select(*keys).reshape(-1)
                    if not all_active:
                        positions = active.reshape(-1).nonzero().flatten()
                        data = data[positions]
                self._pending_environment_steps += environment_steps
                if not data.batch_size.numel():
                    continue
                self._pending_batches.append(data)
                self._pending_steps += data.batch_size.numel()

    def flush_pending_update(self):
        if not self._pending_batches and not getattr(self, "distributed", None):
            self._pending_environment_steps = 0
            return {}
        region = getattr(self, "profile_region", None)
        with region("pending_rollout_concatenation") if region else nullcontext():
            data = torch.cat(self._pending_batches, dim=0) if self._pending_batches else None
        self._pending_batches.clear()
        self._pending_steps = 0
        profile = getattr(self, "profile_call", None)
        metrics = (profile("optimization", optimize_ppo, self, data, device=self.device)
                   if profile else optimize_ppo(self, data))
        if not metrics and getattr(self, "distributed", None):
            self._pending_environment_steps = 0
            return {}
        metrics["train/rollout_agent_samples"] = data.numel() if data is not None else 0
        metrics["train/rollout_steps"] = self._pending_environment_steps
        self._pending_environment_steps = 0
        return metrics
