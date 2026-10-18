"""Validated model specifications and algorithm-independent actor/critic factories."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import List

import torch
import torch.nn as nn

from agents.common.distributions import SquashedGaussianActor
from agents.common.encoders import build_encoder, resolve_encoder_config
from agents.common.network_layers import (
    LEAKY_RELU_NEGATIVE_SLOPE, make_activation, make_mlp, _positive_int, _hidden_dims,
)


class ResidualDenseBlock(nn.Module):
    """Shape-preserving x + F(x), with two dense layers and one activation."""

    def __init__(self, width, activation="leaky_relu"):
        super().__init__()
        self.width = _positive_int(width, "resmlp.width")
        self.branch = nn.Sequential(nn.Linear(width, width), make_activation(activation),
                                    nn.Linear(width, width))

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.width:
            raise ValueError(f"Residual block requires last dimension {self.width}")
        return x + self.branch(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, *, width, num_blocks, activation, input_projection):
        super().__init__()
        self.input_dim = _positive_int(input_dim, "resmlp.input_dim")
        _positive_int(output_dim, "resmlp.output_dim")
        _positive_int(width, "resmlp.width")
        _positive_int(num_blocks, "resmlp.num_blocks")
        if type(input_projection) is not bool:
            raise ValueError("resmlp.input_projection must be a boolean")
        if not input_projection and input_dim != width:
            raise ValueError("resmlp requires input_projection=true when input_dim differs from width")
        self.input_projection = (nn.Sequential(nn.Linear(input_dim, width), make_activation(activation))
                                 if input_projection else nn.Identity())
        self.blocks = nn.Sequential(*(ResidualDenseBlock(width, activation) for _ in range(num_blocks)))
        self.output = nn.Linear(width, output_dim)

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Residual MLP requires last dimension {self.input_dim}")
        return self.output(self.blocks(self.input_projection(x)))


def build_head(input_dim, output_dim, spec):
    """Heads own feature-to-output computation; policy distributions stay outside."""
    _positive_int(input_dim, "head.input_dim")
    _positive_int(output_dim, "head.output_dim")
    if spec["architecture"] == "mlp":
        return make_mlp(input_dim, spec["hidden_dims"], output_dim, spec["activation"])
    if spec["architecture"] == "resmlp":
        return ResidualMLP(input_dim, output_dim, **{key: spec[key] for key in (
            "width", "num_blocks", "activation", "input_projection")})
    raise ValueError(f"Unsupported head architecture: {spec['architecture']!r}")


class Actor(SquashedGaussianActor):
    """Encoded actor head producing Gaussian parameters and exploration scale."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        *, model_spec=None, observation_contract=None,
    ) -> None:
        super().__init__()
        self.action_dim = _positive_int(action_dim, "actor.action_dim")
        spec = model_spec or dict(architecture="mlp", hidden_dims=hidden_dims, activation=activation,
                                  encoder="identity")
        self.encoder = build_encoder(obs_dim, spec["encoder"], observation_contract=observation_contract)
        # Keep net directly on the actor: legacy state keys and RNG order are unchanged.
        self.net = build_head(self.encoder.output_dim, action_dim, spec)
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
        *, model_spec=None, observation_contract=None,
    ) -> None:
        super().__init__()
        spec = model_spec or dict(architecture="mlp", hidden_dims=hidden_dims, activation=activation,
                                  encoder="identity")
        self.encoder = build_encoder(input_dim, spec["encoder"], observation_contract=observation_contract)
        self.net = build_head(self.encoder.output_dim, 1, spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoder(x)).squeeze(-1)


def _resolve_model_spec(value, *, architecture, encoder, hidden_dims, activation, role):
    if not isinstance(value, Mapping):
        raise ValueError(f"network.{role} must be a mapping")
    allowed = {"architecture", "encoder", "hidden_dims", "activation", "observation_layout",
               "width", "num_blocks", "input_projection"}
    if set(value) - allowed:
        raise ValueError(f"Unknown network.{role} settings: {sorted(set(value) - allowed)}")
    kind = value.get("architecture", architecture)
    if not isinstance(kind, str) or kind not in {"mlp", "resmlp"}:
        raise ValueError(f"Unsupported network.{role} architecture: {kind!r}; supported: mlp, resmlp")
    act = value.get("activation", activation)
    make_activation(act)
    encoder = resolve_encoder_config(value.get("encoder", encoder))
    required_layout = "flat" if encoder["type"] == "identity" else "lidar_frenet_v1"
    layout = value.get("observation_layout", required_layout)
    if layout != required_layout:
        raise ValueError(f"network.{role}.observation_layout must be {required_layout!r} for {encoder['type']}")
    spec = dict(architecture=kind, encoder=encoder, activation=act, observation_layout=layout)
    if kind == "mlp":
        if set(value) & {"width", "num_blocks", "input_projection"}:
            raise ValueError(f"network.{role}: width/num_blocks/input_projection require resmlp")
        spec["hidden_dims"] = _hidden_dims(value.get("hidden_dims", hidden_dims), f"network.{role}.hidden_dims")
    else:
        if "hidden_dims" in value:
            raise ValueError(f"network.{role}: resmlp uses width and num_blocks, not hidden_dims")
        spec.update(width=_positive_int(value.get("width", 256), f"network.{role}.width"),
                    num_blocks=_positive_int(value.get("num_blocks", 4), f"network.{role}.num_blocks"),
                    input_projection=value.get("input_projection", True))
        if type(spec["input_projection"]) is not bool:
            raise ValueError(f"network.{role}.input_projection must be a boolean")
    return spec


ROUTING_FIELDS = {"actor": {"distribution", "parameter_sharing"},
                  "critic": {"input_type", "parameter_sharing"}}


def project_network_config(network):
    """Remove application routing fields; retain only meaningful model overrides.

    Inherited settings are fallbacks. Explicit per-role settings take precedence.
    The original mlp Hydra preset projects to its historical four-field payload.
    """
    if not isinstance(network, Mapping):
        raise ValueError("network must be a mapping")
    result = dict(network)
    for role, routing in ROUTING_FIELDS.items():
        if role not in result:
            continue
        if not isinstance(result[role], Mapping):
            raise ValueError(f"network.{role} must be a mapping")
        value = {k: v for k, v in result[role].items() if k not in routing}
        if value.get("architecture") == result.get("architecture", "mlp"):
            value.pop("architecture")
        if value:
            result[role] = value
        else:
            result.pop(role)
    if result.get("encoder") == "identity":
        result.pop("encoder")
    return result


def resolve_network_config(params, *, default_hidden_dims):
    """Produce one canonical spec, retaining the exact legacy MLP representation."""
    network = project_network_config(params.get("network", {}))
    if "spec_version" in network:
        if (type(network["spec_version"]) is not int or network["spec_version"] != 1
                or set(network) != {"spec_version", "architecture", "actor", "critic"}
                or network["architecture"] != "modular"):
            raise ValueError("Unsupported versioned network specification")
        specs = {role: _resolve_model_spec(network[role], architecture="mlp", encoder="identity",
                 hidden_dims=default_hidden_dims, activation="tanh", role=role) for role in ("actor", "critic")}
        for role, spec in specs.items():
            if set(network[role]) != set(spec):
                raise ValueError(f"Versioned network.{role} must contain a complete model specification")
    else:
        allowed = {"architecture", "actor_hidden_dims", "critic_hidden_dims", "activation", "encoder", "actor", "critic"}
        if set(network) - allowed:
            raise ValueError(f"network must be a mapping with fields {sorted(allowed)}")
        architecture = network.get("architecture", "mlp")
        if not isinstance(architecture, str) or architecture not in {"mlp", "resmlp"}:
            raise ValueError(f"Unsupported network architecture: {architecture!r}; supported: mlp, resmlp")
        activation = network.get("activation", params.get("activation", "tanh"))
        make_activation(activation)
        encoder = resolve_encoder_config(network.get("encoder", "identity"))
        specs = {}
        for role, legacy in (("actor", "pi_hidden_dims"), ("critic", "vf_hidden_dims")):
            dims = _hidden_dims(network.get(f"{role}_hidden_dims", params.get(legacy,
                params.get("hidden_dims", default_hidden_dims))), f"network.{role}_hidden_dims")
            specs[role] = _resolve_model_spec(network.get(role, {}), architecture=architecture,
                encoder=encoder, hidden_dims=dims, activation=activation, role=role)
    if params.get("lora") is not None:
        if any(spec["architecture"] != "mlp" for spec in specs.values()):
            raise ValueError("LoRA requires MLP actor and critic; residual MLP + LoRA is unsupported")
        if any(spec["encoder"]["type"] != "identity" for spec in specs.values()):
            raise ValueError("CNN encoders + LoRA are unsupported; current MLP LoRA requires identity encoders")
    if (all(spec["architecture"] == "mlp" and spec["encoder"] == {"type": "identity"}
            for spec in specs.values()) and specs["actor"]["activation"] == specs["critic"]["activation"]):
        return dict(architecture="mlp", actor_hidden_dims=specs["actor"]["hidden_dims"],
                    critic_hidden_dims=specs["critic"]["hidden_dims"], activation=specs["actor"]["activation"])
    return dict(spec_version=1, architecture="modular", **specs)


def model_spec(config, role):
    """Read either the historical MLP config or the versioned per-role spec."""
    if config.get("spec_version") == 1:
        return config[role]
    return dict(architecture=config["architecture"], encoder={"type": "identity"},
                hidden_dims=config[f"{role}_hidden_dims"], activation=config["activation"],
                observation_layout="flat")


def build_actor(obs_dim, action_dim, config, *, log_std_init=0.0,
                observation_contract=None, observation_contracts=None):
    """Return an actor mapping observations to (mean, scale)."""
    spec = model_spec(config, "actor")
    if not math.isfinite(log_std_init) or not Actor.LOG_STD_MIN <= log_std_init <= Actor.LOG_STD_MAX:
        raise ValueError("log_std_init must be within the actor's log standard deviation bounds")
    if observation_contracts is not None:
        observation_contract = next(iter(observation_contracts.values()))
        if spec["encoder"]["type"] != "identity":
            from agents.common.observations import observation_segments
            layouts = [observation_segments(contract, obs_dim) for contract in observation_contracts.values()]
            if any(layout != layouts[0] for layout in layouts[1:]):
                raise ValueError("CNN actors require identical observation segments across learners")
    actor = Actor(obs_dim, action_dim, spec.get("hidden_dims", []), spec["activation"],
                  model_spec=spec, observation_contract=observation_contract)
    with torch.no_grad():
        actor.log_std.fill_(log_std_init)
    return actor


def build_critic(input_dim, config, *, observation_contract=None):
    """Return a critic mapping local observations or global states to values."""
    spec = model_spec(config, "critic")
    return Critic(input_dim, spec.get("hidden_dims", []), spec["activation"],
                  model_spec=spec, observation_contract=observation_contract)


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


def expand_observation_inputs(state, expected, config, *, source_width, expand_inputs, role="actor",
                              new_parameter_prefixes=()):
    """Expand a validated feature prefix and retain newly initialized branches.

    The head specification identifies its input projection. Everything else
    must match exactly; callers validate the complete prepared state before use.
    """
    result = dict(state)
    if new_parameter_prefixes:
        added = {key for key in expected if key.startswith(new_parameter_prefixes)}
        if not added or set(result) != set(expected) - added:
            raise ValueError("Incompatible checkpoint actor: unexpected source or new-branch state keys")
        result.update({key: expected[key].clone() for key in added})
    if not expand_inputs:
        return result
    spec = model_spec(config, role)
    if spec["architecture"] == "mlp":
        key = "net.0.weight"
    elif spec["architecture"] == "resmlp" and spec["input_projection"]:
        key = "net.input_projection.0.weight"
    else:
        raise ValueError("Observation expansion requires a compatible MLP or residual MLP input projection")
    weight, target = result.get(key), expected.get(key)
    if (not isinstance(target, torch.Tensor) or target.ndim != 2
            or not isinstance(weight, torch.Tensor) or weight.shape != (target.shape[0], source_width)
            or weight.dtype != target.dtype or source_width >= target.shape[1]):
        raise ValueError("First-layer weights do not match the observation-prefix dimensions")
    expanded = torch.zeros_like(target)
    expanded[:, :source_width] = weight
    result[key] = expanded
    return result
