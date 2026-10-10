"""TorchRL bridge for the single-agent Gymnasium task adapter."""
from torchrl.envs import GymWrapper, set_gym_backend

from adapters.gymnasium import RaceGymEnv


class RaceTorchRLEnv(GymWrapper):
    """Use the known Gymnasium backend without probing unrelated integrations."""

    def __init__(self, env: RaceGymEnv, **kwargs):
        if not isinstance(env, RaceGymEnv):
            raise TypeError("RaceTorchRLEnv requires a single-agent RaceGymEnv")
        with set_gym_backend("gymnasium"):
            super().__init__(env, **kwargs)

    @staticmethod
    def get_library_name(env):
        return "gymnasium"

    def _build_env(self, env, from_pixels=False, pixels_only=False):
        # Generic pixel-wrapper detection also imports legacy Gym. Task observations
        # are composed vectors; rendering does not turn them into pixel observations.
        if from_pixels or pixels_only:
            raise ValueError("RaceTorchRLEnv requires composed vector observations")
        self.batch_size = self._get_batch_size(env)
        self.from_pixels = self.pixels_only = False
        return env

    @property
    def _is_batched(self):
        # RaceGymEnv wraps one task, so SB3/IsaacLab vector detection is unnecessary.
        return False
