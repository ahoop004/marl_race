"""Feature encoder contract independent of algorithm losses and network heads."""
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn


class FeatureEncoder(nn.Module, ABC):
    """Encode observations [..., input_dim] as features [..., output_dim].

    Leading dimensions belong to callers (time, environments, or agents).
    Future encoders can implement this contract without changing objectives.
    Stateful/recurrent encoders will additionally need explicit TensorDict
    state keys and a compatible collector; those combinations are unavailable.
    """
    input_dim: int
    output_dim: int

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityEncoder(FeatureEncoder):
    def __init__(self, input_dim):
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("Encoder input_dim must be a positive integer")
        self.input_dim = self.output_dim = input_dim

    def forward(self, observations):
        if observations.ndim == 0 or observations.shape[-1] != self.input_dim:
            raise ValueError(f"Expected observation width {self.input_dim}, got shape {tuple(observations.shape)}")
        return observations


def resolve_encoder_config(config="identity"):
    """Validate encoder settings without constructing a module or consuming RNG."""
    if isinstance(config, str):
        config = {"type": config}
    if not isinstance(config, Mapping) or set(config) != {"type"}:
        raise ValueError("encoder must be an identifier or a mapping with field 'type'")
    if config["type"] != "identity":
        raise ValueError(f"Unsupported observation encoder: {config['type']!r}; only the identity observation encoder is implemented")
    return dict(config)


def build_encoder(input_dim, kind="identity"):
    kind = resolve_encoder_config(kind)["type"]
    if kind != "identity":
        raise ValueError(f"Unsupported observation encoder: {kind!r}")
    return IdentityEncoder(input_dim)
