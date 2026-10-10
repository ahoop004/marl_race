"""PPO policy setup, inference and the existing checkpoint format."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from agents.common.outputs import PolicyOutput
from agents.common.networks import Actor, Critic
from utils.torch_io import resolve_device


class PPOPolicy:
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
        self.lr = float(params.get("learning_rate", 3e-4))
        self.lr_schedule = str(params.get("lr_schedule", "constant"))
        if self.lr_schedule not in {"constant", "linear"}:
            raise ValueError("PPO lr_schedule must be 'constant' or 'linear'.")
        if self.lr_schedule == "linear" and "learning_rate_end" not in params:
            raise ValueError("Linear PPO lr_schedule requires learning_rate_end.")
        self.lr_end = float(params.get("learning_rate_end", self.lr))
        if not all(np.isfinite(rate) and rate > 0 for rate in (self.lr, self.lr_end)):
            raise ValueError("PPO learning rates must be finite and positive.")
        if self.lr_schedule == "constant" and self.lr_end != self.lr:
            raise ValueError("A different learning_rate_end requires lr_schedule: linear.")
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
        self.target_kl = params.get("target_kl")
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)
            if not np.isfinite(self.target_kl) or self.target_kl <= 0:
                raise ValueError("target_kl must be finite and positive")

        hidden_dims: List[int] = list(
            params.get("pi_hidden_dims", params.get("hidden_dims", [64, 64]))
        )
        vf_dims: List[int] = list(
            params.get("vf_hidden_dims", params.get("hidden_dims", [64, 64]))
        )
        activation: str = str(params.get("activation", "tanh"))
        self.actor_hidden_dims = list(hidden_dims)
        self.critic_hidden_dims = list(vf_dims)
        self.activation = activation

        device_str = str(params.get("device", "cpu"))
        self.device = resolve_device([device_str])

        self.actor = Actor(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        log_std_init = float(params.get("log_std_init", 0.0))
        if not np.isfinite(log_std_init) or not self.actor.LOG_STD_MIN <= log_std_init <= self.actor.LOG_STD_MAX:
            raise ValueError("log_std_init must be within the actor's log standard deviation bounds")
        with torch.no_grad():
            self.actor.log_std.fill_(log_std_init)
        self.critic = Critic(obs_dim, vf_dims, activation).to(self.device)
        self._optim_parameters = tuple(self.actor.parameters()) + tuple(self.critic.parameters())
        self.optimizer = optim.Adam(self._optim_parameters, lr=self.lr)

    def set_training_progress(self, progress: float) -> None:
        """Set LR from the trainer's globally completed budget fraction.

        Evaluation never advances this schedule. In parallel training only
        the parent optimizer receives progress, not individual collectors.
        """
        if self.lr_schedule == "linear":
            fraction = float(np.clip(progress, 0.0, 1.0))
            rate = self.lr + fraction * (self.lr_end - self.lr)
            for group in self.optimizer.param_groups:
                group["lr"] = rate

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
        return torch.tanh(self.actor.net(obs_t))[0].cpu().numpy()

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
        if observation_extension:
            for name, module in (("actor", self.actor), ("critic", self.critic)):
                weights = ckpt[name]["net.0.weight"]
                expected = module.state_dict()["net.0.weight"]
                if weights.shape[0] == expected.shape[0] and weights.shape[1] < expected.shape[1]:
                    ckpt[name]["net.0.weight"] = torch.nn.functional.pad(
                        weights, (0, expected.shape[1] - weights.shape[1]))
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

