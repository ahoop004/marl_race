"""Dense layers and activations shared by observation encoders and model heads."""
from functools import partial
from typing import List, Optional

import torch.nn as nn

# Fixed as part of the "leaky_relu" checkpoint activation contract.
LEAKY_RELU_NEGATIVE_SLOPE = 0.2


def make_activation(activation):
    act_map = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "swish": nn.SiLU,
        "leaky_relu": partial(nn.LeakyReLU, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE),
    }
    if not isinstance(activation, str) or activation.lower() not in act_map:
        raise ValueError(f"Unsupported network activation: {activation!r}")
    return act_map[activation.lower()]()


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _hidden_dims(value, field):
    if not isinstance(value, (list, tuple)) or any(type(d) is not int or d <= 0 for d in value):
        raise ValueError(f"{field} must be a sequence of positive integers")
    return list(value)


def make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a fully-connected MLP."""
    make_activation(activation)  # Validate even a linear-only head.

    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), make_activation(activation)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))

    if output_activation:
        layers.append(make_activation(output_activation))

    return nn.Sequential(*layers)

