"""Feature encoder contract independent of algorithm losses and network heads."""
from abc import ABC, abstractmethod

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
        self.input_dim = self.output_dim = input_dim

    def forward(self, observations):
        if observations.shape[-1] != self.input_dim:
            raise ValueError(f"Expected observation width {self.input_dim}, got {observations.shape[-1]}")
        return observations


def build_encoder(input_dim, kind="identity"):
    if kind != "identity":
        raise ValueError(f"Unsupported observation encoder: {kind!r}")
    return IdentityEncoder(input_dim)
