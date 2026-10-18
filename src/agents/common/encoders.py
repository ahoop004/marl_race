"""Feature encoder contract independent of algorithm losses and network heads."""
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

from agents.common.network_layers import make_activation, make_mlp, _positive_int, _hidden_dims
from agents.common.observations import observation_segments


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


class ResidualConv1DBlock(nn.Module):
    """Two convolutions with a projected shortcut at channel/stride transitions."""

    def __init__(self, input_channels, output_channels, kernel_size=5, stride=1, activation="leaky_relu"):
        super().__init__()
        self.input_channels = _positive_int(input_channels, "rescnn.input_channels")
        _positive_int(output_channels, "rescnn.output_channels")
        _positive_int(kernel_size, "rescnn.kernel_size")
        _positive_int(stride, "rescnn.stride")
        if kernel_size % 2 != 1:
            raise ValueError("Residual CNN kernel_size must be odd for compatible shortcut lengths")
        self.branch = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            make_activation(activation),
            nn.Conv1d(output_channels, output_channels, kernel_size, padding=kernel_size // 2),
        )
        self.shortcut = (nn.Identity() if input_channels == output_channels and stride == 1
                         else nn.Conv1d(input_channels, output_channels, 1, stride=stride))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.input_channels or x.shape[2] == 0:
            raise ValueError(f"Residual CNN requires [batch, {self.input_channels}, positive scan length]")
        return self.shortcut(x) + self.branch(x)


def fusion_feature_segments(config, segments):
    """Feature order shared by fusion forward calls and actor-head transfer."""
    lidar_end = config["lidar_features"]
    driving_end = lidar_end + config["vehicle_features"]
    neighbor_width = config["neighbor_features"] if segments["neighbors"][1] > segments["neighbors"][0] else 0
    return {"lidar": [0, lidar_end], "vehicle_track": [lidar_end, driving_end],
            "neighbors": [driving_end, driving_end + neighbor_width]}


class CNN1DFusionEncoder(FeatureEncoder):
    """Ordered LiDAR convolutions fused with separate vehicle and neighbor MLPs.

    Shared branch names make the driving representation independently reusable.
    Residual and ordinary CNNs differ only in the LiDAR backbone construction.
    """

    def __init__(self, input_dim, config, observation_contract):
        super().__init__()
        self.input_dim = _positive_int(input_dim, "encoder.input_dim")
        self.segments = observation_segments(observation_contract, input_dim)
        self.config = resolve_encoder_config(config)
        activation = self.config["activation"]
        channels = 1
        stages = []
        for width in self.config["channels"]:
            if self.config["type"] == "rescnn1d_fusion":
                for block in range(self.config["num_blocks"]):
                    stages.extend([ResidualConv1DBlock(channels, width, self.config["kernel_size"],
                        self.config["stride"] if block == 0 else 1, activation), make_activation(activation)])
                    channels = width
            else:
                stages.extend([nn.Conv1d(channels, width, self.config["kernel_size"],
                    stride=self.config["stride"], padding=self.config["kernel_size"] // 2), make_activation(activation)])
                channels = width
        self.lidar_backbone = nn.Sequential(*stages)
        self.lidar_projection = nn.Sequential(nn.AdaptiveAvgPool1d(self.config["pool_bins"]), nn.Flatten(1),
            nn.Linear(channels * self.config["pool_bins"], self.config["lidar_features"]), make_activation(activation))
        start, stop = self.segments["vehicle_track"]
        self.vehicle_track = make_mlp(stop - start, self.config["vehicle_hidden_dims"],
                                      self.config["vehicle_features"], activation, activation)
        start, stop = self.segments["neighbors"]
        self.neighbors = (make_mlp(stop - start, self.config["neighbor_hidden_dims"],
                          self.config["neighbor_features"], activation, activation) if stop > start else None)
        self.feature_segments = fusion_feature_segments(self.config, self.segments)
        self.output_dim = self.feature_segments["neighbors"][1]

    def forward(self, observations):
        if observations.ndim == 0 or observations.shape[-1] != self.input_dim:
            raise ValueError(f"Expected observation width {self.input_dim}, got shape {tuple(observations.shape)}")
        leading = observations.shape[:-1]
        flat = observations.reshape(-1, self.input_dim)
        start, stop = self.segments["lidar"]
        lidar = self.lidar_projection(self.lidar_backbone(flat[:, start:stop].unsqueeze(1)))
        start, stop = self.segments["vehicle_track"]
        features = [lidar, self.vehicle_track(flat[:, start:stop])]
        if self.neighbors is not None:
            start, stop = self.segments["neighbors"]
            features.append(self.neighbors(flat[:, start:stop]))
        return torch.cat(features, dim=-1).reshape(*leading, self.output_dim)


def resolve_encoder_config(config="identity"):
    """Validate and fully specify encoder settings without consuming RNG."""
    if isinstance(config, str):
        config = {"type": config}
    if not isinstance(config, Mapping) or "type" not in config:
        raise ValueError("encoder must be an identifier or a mapping with field 'type'")
    kind = config["type"]
    if not isinstance(kind, str) or kind not in {"identity", "cnn1d_fusion", "rescnn1d_fusion"}:
        raise ValueError(f"Unsupported observation encoder: {kind!r}; supported: identity, cnn1d_fusion, rescnn1d_fusion")
    if kind == "identity":
        if set(config) != {"type"}:
            raise ValueError("identity encoder accepts only the type field")
        return dict(config)
    defaults = dict(channels=[32, 64, 128], kernel_size=5, stride=2, pool_bins=8,
                    lidar_features=128, vehicle_hidden_dims=[64], vehicle_features=64,
                    neighbor_hidden_dims=[32], neighbor_features=32, activation="leaky_relu")
    if kind == "rescnn1d_fusion":
        defaults["num_blocks"] = 2  # Blocks per convolutional stage.
    unknown = set(config) - {"type", *defaults}
    if unknown:
        raise ValueError(f"Unknown {kind} encoder settings: {sorted(unknown)}")
    resolved = {"type": kind, **defaults, **config}
    for field in ("channels", "vehicle_hidden_dims", "neighbor_hidden_dims"):
        resolved[field] = _hidden_dims(resolved[field], f"encoder.{field}")
    if not resolved["channels"]:
        raise ValueError("encoder.channels must contain at least one convolutional stage")
    for field in ("kernel_size", "stride", "pool_bins", "lidar_features", "vehicle_features", "neighbor_features"):
        _positive_int(resolved[field], f"encoder.{field}")
    if resolved["kernel_size"] % 2 != 1:
        raise ValueError("encoder.kernel_size must be odd")
    if kind == "rescnn1d_fusion":
        _positive_int(resolved["num_blocks"], "encoder.num_blocks")
    make_activation(resolved["activation"])
    return resolved


def build_encoder(input_dim, kind="identity", *, observation_contract=None):
    config = resolve_encoder_config(kind)
    if config["type"] == "identity":
        return IdentityEncoder(input_dim)
    return CNN1DFusionEncoder(input_dim, config, observation_contract)
