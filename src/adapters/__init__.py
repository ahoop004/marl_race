"""Optional external interoperability adapters, loaded only when requested."""
from adapters.rewards import RewardMapping

__all__ = ["RaceGymEnv", "RaceParallelEnv", "NativeRaceTorchRLEnv", "RewardMapping"]


def __getattr__(name):
    if name == "RaceGymEnv":
        from adapters.gymnasium import RaceGymEnv
        return RaceGymEnv
    if name == "RaceParallelEnv":
        from adapters.pettingzoo import RaceParallelEnv
        return RaceParallelEnv
    if name == "NativeRaceTorchRLEnv":
        from adapters.native_torchrl import NativeRaceTorchRLEnv
        return NativeRaceTorchRLEnv
    raise AttributeError(name)
