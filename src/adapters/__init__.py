"""Native TorchRL environment and task reward translation."""
from adapters.rewards import RewardMapping

__all__ = ["NativeRaceTorchRLEnv", "RewardMapping"]


def __getattr__(name):
    if name == "NativeRaceTorchRLEnv":
        from adapters.native_torchrl import NativeRaceTorchRLEnv
        return NativeRaceTorchRLEnv
    raise AttributeError(name)
