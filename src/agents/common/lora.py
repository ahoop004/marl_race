"""Low-rank residuals on a frozen driving actor, shared or routed by teammate."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from agents.common.networks import Actor, ResidualMLP, build_head
from agents.common.encoders import IdentityEncoder
from agents.common.distributions import SquashedGaussianActor


def resolve_lora_config(value):
    """None preserves ordinary full fine-tuning; reject misspelled adapter options."""
    if value is None:
        return None
    allowed = {"mode", "rank", "alpha", "train_log_std", "per_agent_log_std", "target"}
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError(f"lora must be a mapping with fields {sorted(allowed)}")
    if value.get("target", "actor_head") != "actor_head":
        raise ValueError("lora.target must be actor_head; encoder and convolutional LoRA are unsupported")
    # actor_head is the historical behavior. Omit the default from the canonical
    # options so legacy contracts and state layouts remain exactly compatible.
    mode = value.get("mode", "shared")
    if not isinstance(mode, str) or mode not in {"shared", "per_agent"}:
        raise ValueError("lora.mode must be shared or per_agent")
    rank = value.get("rank", 4)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("lora.rank must be a positive integer")
    alpha = value.get("alpha", rank)
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("lora.alpha must be finite and positive")
    train_log_std = value.get("train_log_std", True)
    if not isinstance(train_log_std, bool):
        raise ValueError("lora.train_log_std must be a boolean")
    per_agent_log_std = value.get("per_agent_log_std", False)
    if not isinstance(per_agent_log_std, bool) or (per_agent_log_std and mode != "per_agent"):
        raise ValueError("lora.per_agent_log_std requires boolean true with mode=per_agent")
    config = dict(mode=mode, rank=rank, alpha=float(alpha), train_log_std=train_log_std)
    if per_agent_log_std:
        config["per_agent_log_std"] = True
    return config


class LowRankResidual(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        if rank > min(in_features, out_features):
            raise ValueError("lora.rank cannot exceed either adapted layer dimension")
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))
        self.scale = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, inputs):
        return F.linear(F.linear(inputs, self.A), self.B) * self.scale


def head_adapter_targets(head):
    """Discover named hidden linear modules, preserving historical MLP indices."""
    if isinstance(head, nn.Sequential):
        return {str(i): layer for i, layer in enumerate(head)
                if isinstance(layer, nn.Linear) and i != len(head) - 1}
    if isinstance(head, ResidualMLP):
        return {name: module for name, module in head.named_modules()
                if isinstance(module, nn.Linear) and name != "output"}
    raise ValueError("LoRA actor_head supports MLP and residual MLP heads only")


def validate_lora_spec(spec, config, *, input_dim=None):
    """Validate all target dimensions at configuration time using storage-free heads."""
    config = resolve_lora_config(config)
    if input_dim is None:
        input_dim = spec["width"] if spec["architecture"] == "resmlp" and not spec["input_projection"] else config["rank"]
    with torch.device("meta"):
        targets = head_adapter_targets(build_head(input_dim, 1, spec))
    if not targets:
        raise ValueError("LoRA requires at least one actor hidden layer or residual branch")
    for name, layer in targets.items():
        if config["rank"] > min(layer.in_features, layer.out_features):
            raise ValueError(f"lora.rank exceeds adapted module net.{name} dimensions ({layer.in_features}, {layer.out_features})")


def parameter_accounting(policy):
    """Count disjoint actor ownership groups alongside the fresh critic."""
    def count(module, trainable=False):
        return sum(p.numel() for p in module.parameters() if not trainable or p.requires_grad) if module is not None else 0
    actor = policy.actor
    total, trainable = count(actor), count(actor, True)
    exploration = sum(p.numel() for name, p in actor.named_parameters()
                      if p.requires_grad and (name == "log_std" or name.startswith("log_stds.")))
    return dict(actor_total=total, actor_frozen=total - trainable, actor_trainable=trainable,
                lora_trainable=count(getattr(actor, "adapters", None), True),
                neighbor_encoder_trainable=count(getattr(getattr(actor, "encoder", None), "neighbors", None), True),
                exploration_trainable=exploration, critic_trainable=count(policy.critic, True))


class LoRAActor(SquashedGaussianActor):
    """Frozen encoded head with shared or per-agent low-rank residual banks.

    The driving encoder and base head are shared and frozen. A CNN neighbor
    encoder is shared and trainable; independence applies to adapter banks and
    optional exploration vectors, not to the neighbor encoder.
    """

    def __init__(self, base: Actor, config, n_agents, input_dims=None):
        super().__init__()
        if not isinstance(base, Actor):
            raise ValueError("LoRA requires a modular Actor")
        self.action_dim = base.action_dim
        self.net = base.net
        self.log_std = base.log_std
        self.encoder = base.encoder
        self.config = resolve_lora_config(config)
        self.legacy = isinstance(self.encoder, IdentityEncoder) and isinstance(self.net, nn.Sequential)
        targets = head_adapter_targets(self.net)
        if not targets:
            raise ValueError("LoRA requires at least one actor hidden layer or residual branch")
        self.target_layers = [int(name) for name in targets] if isinstance(self.net, nn.Sequential) else list(targets)
        self.target_modules = [f"net.{name}" for name in targets]
        self._adapter_keys = {name: name.replace(".", "__") for name in targets}
        count = n_agents if self.config["mode"] == "per_agent" else 1
        raw_inputs = list(input_dims or [self.encoder.input_dim] * count)
        if len(raw_inputs) != count:
            raise ValueError("LoRA input dimensions must match adapter routing")
        if not self.legacy and any(width != self.encoder.input_dim for width in raw_inputs):
            raise ValueError("Modular head LoRA requires identical local observation widths")
        self.input_dims = raw_inputs if self.legacy else [self.encoder.output_dim] * count
        first = next(iter(targets))
        self.adapters = nn.ModuleList([
            nn.ModuleDict({self._adapter_keys[name]: LowRankResidual(
                self.input_dims[bank] if name == first else layer.in_features, layer.out_features,
                self.config["rank"], self.config["alpha"],
            ) for name, layer in targets.items()}) for bank in range(count)
        ])
        self.net.requires_grad_(False)
        self.encoder.requires_grad_(False)
        if getattr(self.encoder, "neighbors", None) is not None:
            self.encoder.neighbors.requires_grad_(True)
        self.log_std.requires_grad_(self.config["train_log_std"] and not self.config.get("per_agent_log_std"))
        if self.config.get("per_agent_log_std"):
            self.log_stds = nn.ParameterList([
                nn.Parameter(self.log_std.detach().clone(), requires_grad=self.config["train_log_std"])
                for _ in range(count)
            ])

    def adaptation_contract(self):
        if self.legacy:
            return {}
        return {"version": 2, "target": "actor_head", "target_modules": self.target_modules,
                "freezing": {"driving_encoder": True, "base_head": True,
                             "neighbor_encoder": False, "neighbor_encoder_shared": True}}

    def base_state_dict(self):
        return {key: value for key, value in self.state_dict().items()
                if not key.startswith(("adapters.", "log_stds."))}

    def reset_adapters(self):
        for bank in self.adapters:
            for residual in bank.values():
                residual.reset_parameters()

    def _sequence(self, sequence, inputs, bank, prefix):
        if isinstance(sequence, nn.Identity):
            return sequence(inputs)
        for name, layer in sequence.named_children():
            path = f"{prefix}.{name}"
            output = layer(inputs)
            key = self._adapter_keys.get(path)
            if key is not None:
                output = output + bank[key](inputs)
            inputs = output
        return inputs

    def _mean(self, obs, bank):
        if isinstance(self.net, nn.Sequential):
            # Keep the historical MLP arithmetic and adapter keys unchanged.
            for i, layer in enumerate(self.net):
                output = layer(obs)
                if str(i) in bank:
                    residual = bank[str(i)]
                    output = output + residual(obs[..., :residual.A.shape[1]])
                obs = output
            return obs
        obs = self._sequence(self.net.input_projection, obs, bank, "input_projection")
        for index, block in enumerate(self.net.blocks):
            obs = obs + self._sequence(block.branch, obs, bank, f"blocks.{index}.branch")
        return self.net.output(obs)

    def forward(self, obs, adapter_indices=None):
        obs = self.encoder(obs)
        if self.config["mode"] == "shared":
            if adapter_indices is not None:
                raise ValueError("Shared LoRA does not take per-agent adapter indices")
            mean = self._mean(obs, self.adapters[0])
        else:
            if (adapter_indices is None or adapter_indices.shape != (len(obs),)
                    or adapter_indices.dtype != torch.long):
                raise ValueError("Per-agent LoRA requires one integer adapter index per observation")
            if torch.any((adapter_indices < 0) | (adapter_indices >= len(self.adapters))):
                raise ValueError("Invalid LoRA adapter index")
            mean = obs.new_zeros((len(obs), self.action_dim))
            for index, bank in enumerate(self.adapters):
                rows = torch.nonzero(adapter_indices == index, as_tuple=True)[0]
                if rows.numel():
                    mean = mean.index_copy(0, rows, self._mean(obs.index_select(0, rows), bank))
        if self.config.get("per_agent_log_std"):
            std = obs.new_zeros(mean.shape)
            for index, log_std in enumerate(self.log_stds):
                rows = torch.nonzero(adapter_indices == index, as_tuple=True)[0]
                if rows.numel():
                    values = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
                    std = std.index_copy(0, rows, values.expand(len(rows), -1))
            return mean, std
        return mean, self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
