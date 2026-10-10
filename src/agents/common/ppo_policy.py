"""PPO policy setup, inference and the existing checkpoint format."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from agents.common.outputs import PolicyOutput
from agents.common.optimization import OnPolicyOptimizationSettings
from agents.common.networks import build_actor, build_critic, resolve_network_config
from agents.common.checkpoints import (
    restore_policy_state, transfer_network_state, validate_network_checkpoint,
)
from utils.torch_io import resolve_device


class PPOPolicy(OnPolicyOptimizationSettings):
    """Actor/critic state shared by TorchRL training and checkpoint evaluation."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        params: Dict,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = len(self.action_low)
        self.action_contract = dict(params.get("_action_contract", {"speed_control": "direct"}))
        self.physics_contract = params.get("_physics_contract")
        self.observation_contract = params.get("_observation_contract")

        # Hyperparameters (merged from training_defaults + scenario params)
        self.configure_optimization(params)
        self.gamma = float(params.get("gamma", 0.99))
        self.gae_lambda = float(params.get("gae_lambda", 0.95))
        self.clip_range = float(params.get("clip_range", 0.2))
        self.ent_coef = float(params.get("ent_coef", 0.01))
        self.vf_coef = float(params.get("vf_coef", 0.5))
        self.max_grad_norm = float(params.get("max_grad_norm", 0.5))
        self.n_steps = int(params.get("n_steps", 2048))
        self.n_epochs = int(params.get("n_epochs", 10))
        self.batch_size = int(params.get("batch_size", 64))
        self.min_rollout_steps = int(params.get("min_rollout_steps", 1))
        if self.min_rollout_steps < 1:
            raise ValueError("min_rollout_steps must be positive")
        self.network_config = resolve_network_config(params, default_hidden_dims=[64, 64])
        self.actor_hidden_dims = self.network_config["actor_hidden_dims"]
        self.critic_hidden_dims = self.network_config["critic_hidden_dims"]
        self.activation = self.network_config["activation"]

        device_str = str(params.get("device", "cpu"))
        self.device = resolve_device([device_str])

        self.actor = build_actor(
            obs_dim, self.action_dim, self.network_config,
            log_std_init=float(params.get("log_std_init", 0.0)),
        ).to(self.device)
        self.critic = build_critic(obs_dim, self.network_config).to(self.device)
        self._optim_parameters = tuple(self.actor.parameters()) + tuple(self.critic.parameters())
        self.optimizer = optim.Adam(self._optim_parameters, lr=self.lr)

    @torch.no_grad()
    def value_batch(self, observations: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return self.critic(obs).cpu().numpy()

    def value(self, observation: np.ndarray) -> float:
        return float(self.value_batch(np.asarray(observation)[None])[0])

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic evaluation action without evaluating the critic."""
        obs_t = torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32, device=self.device)
        actions, _ = self.actor.get_action(obs_t, deterministic=True)
        return actions[0].cpu().numpy()

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action from policy.

        Returns:
            (action_normalized, log_prob, value)
            action_normalized is in [-1, 1] — caller denormalizes for env.step()
        """
        actions, log_probs, values = self.act_batch(np.asarray(obs)[None], deterministic)
        return actions[0], float(log_probs[0]), float(values[0])

    @torch.no_grad()
    def sample_batch(self, observations: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions, log_probs, raw_actions = self.actor.get_action(
            obs_t, deterministic=deterministic, return_raw=True
        )
        values = self.critic(obs_t)
        # One device-to-host transfer for all environments and policy outputs.
        outputs = torch.cat((actions, raw_actions, log_probs[:, None], values[:, None]), dim=1).cpu().numpy()
        return PolicyOutput(outputs[:, :self.action_dim], outputs[:, -2],
                            outputs[:, self.action_dim:2 * self.action_dim].copy(), outputs[:, -1])

    def act_batch(self, observations, deterministic=False):
        output = self.sample_batch(observations, deterministic)
        return output.actions, output.log_probs, output.values

    def evaluation_actions(self, agent_ids, observations):
        if len(agent_ids) != 1:
            raise ValueError("PPO inference requires one active policy agent")
        aid = agent_ids[0]
        return {aid: self.predict(observations[aid])}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "algorithm": "ppo",
                "network": self.network_config,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "action_contract": self.action_contract,
                "physics_contract": self.physics_contract,
                "observation_contract": self.observation_contract,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "activation": self.activation,
            },
            path,
        )

    def load(self, path: str, *, load_optimizer: bool = True, observation_extension: Optional[str] = None) -> None:
        """Load actor/critic weights; transfer runs keep their fresh optimizer."""
        from utils.torch_io import safe_load
        ckpt = safe_load(path, map_location=self.device)
        validate_network_checkpoint(ckpt, self.network_config)
        actor_state = ckpt.get("actor")
        if observation_extension:
            actor_state = transfer_network_state(
                actor_state, self.actor.state_dict(), self.network_config, expand_inputs=True)
            ckpt = {**ckpt, "critic": transfer_network_state(
                ckpt.get("critic"), self.critic.state_dict(), self.network_config, expand_inputs=True)}
        restore_policy_state(ckpt, self.actor, self.critic, self.optimizer,
                             actor_state=actor_state, load_optimizer=load_optimizer)

