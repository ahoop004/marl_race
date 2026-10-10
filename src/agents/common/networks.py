"""MLP networks and the small actor/critic construction factory."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import math
from typing import List, Optional

import torch
import torch.nn as nn

from agents.common.distributions import SquashedGaussianActor
from agents.common.encoders import build_encoder

# Fixed as part of the "leaky_relu" checkpoint activation contract.
LEAKY_RELU_NEGATIVE_SLOPE = 0.2


def make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a fully-connected MLP."""
    act_map = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "swish": nn.SiLU,
        "leaky_relu": partial(nn.LeakyReLU, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE),
    }
    if activation.lower() not in act_map:
        raise ValueError(f"Unsupported network activation: {activation!r}")
    Act = act_map[activation.lower()]

    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), Act()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))

    if output_activation:
        OutAct = act_map.get(output_activation.lower())
        if OutAct:
            layers.append(OutAct())

    return nn.Sequential(*layers)


class Actor(SquashedGaussianActor):
    """MLP producing Gaussian parameters; retain legacy net/log_std keys."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.encoder = build_encoder(obs_dim)
        # net is the actor head; its names retain the existing checkpoint layout.
        self.net = make_mlp(self.encoder.output_dim, hidden_dims, action_dim, activation)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(self.encoder(obs))
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std.exp()


class Critic(nn.Module):
    """Value function — maps obs (or global state for MAPPO) to scalar.

    input_dim=obs_dim  for PPO (local obs)
    input_dim=global_state_dim+n_trainable_agents for agent-conditioned MAPPO
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(input_dim)
        self.net = make_mlp(self.encoder.output_dim, hidden_dims, 1, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoder(x)).squeeze(-1)


def resolve_network_config(params, *, default_hidden_dims):
    """Resolve explicit network settings, with historical params as defaults."""
    network = params.get("network", {})
    allowed = {"architecture", "actor_hidden_dims", "critic_hidden_dims", "activation"}
    if not isinstance(network, Mapping) or set(network) - allowed:
        raise ValueError(f"network must be a mapping with fields {sorted(allowed)}")
    config = {
        "architecture": "mlp",
        "actor_hidden_dims": list(params.get("pi_hidden_dims", params.get("hidden_dims", default_hidden_dims))),
        "critic_hidden_dims": list(params.get("vf_hidden_dims", params.get("hidden_dims", default_hidden_dims))),
        "activation": str(params.get("activation", "tanh")),
        **network,
    }
    if config["architecture"] != "mlp":
        raise ValueError(f"Unsupported network architecture: {config['architecture']!r}; only mlp is available")
    for field in ("actor_hidden_dims", "critic_hidden_dims"):
        dims = config[field]
        if not isinstance(dims, (list, tuple)) or any(
                isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in dims):
            raise ValueError(f"network.{field} must be a sequence of positive integers")
        config[field] = list(dims)
    if not isinstance(config["activation"], str):
        raise ValueError("network.activation must be a string")
    if config["activation"].lower() not in {"relu", "tanh", "silu", "swish", "leaky_relu"}:
        raise ValueError(f"Unsupported network activation: {config['activation']!r}")
    return config


def build_actor(obs_dim, action_dim, config, *, log_std_init=0.0):
    """Return an actor mapping observations to (mean, scale)."""
    if config["architecture"] != "mlp":
        raise ValueError(f"Unsupported actor architecture: {config['architecture']!r}")
    actor = Actor(obs_dim, action_dim, config["actor_hidden_dims"], config["activation"])
    if not math.isfinite(log_std_init) or not actor.LOG_STD_MIN <= log_std_init <= actor.LOG_STD_MAX:
        raise ValueError("log_std_init must be within the actor's log standard deviation bounds")
    with torch.no_grad():
        actor.log_std.fill_(log_std_init)
    return actor


def build_critic(input_dim, config):
    """Return a critic mapping local observations or global states to values."""
    if config["architecture"] != "mlp":
        raise ValueError(f"Unsupported critic architecture: {config['architecture']!r}")
    return Critic(input_dim, config["critic_hidden_dims"], config["activation"])


def route_actor(actor, agent_ids, *, actor_mode="shared", lora_config=None, input_dims=None):
    """Apply routing after critic construction to preserve seeded initialization."""
    if actor_mode not in {"shared", "independent"}:
        raise ValueError("actor_mode must be shared or independent")
    if actor_mode == "independent":
        if lora_config is not None:
            raise ValueError("Independent actors cannot also use LoRA")
        from agents.common.independent import IndependentActors
        return IndependentActors(actor, agent_ids)
    if lora_config is not None:
        from agents.common.lora import LoRAActor
        return LoRAActor(actor, lora_config, len(agent_ids), input_dims)
    return actor
