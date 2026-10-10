"""Framework-independent task contracts and resolved episode metadata."""
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    agent_ids: tuple[str, ...]
    observation_dims: Mapping[str, int]
    observation_contracts: Mapping[str, Any]
    action_lows: Mapping[str, np.ndarray]
    action_highs: Mapping[str, np.ndarray]
    state_dim: int
    state_version: str


@dataclass(frozen=True)
class EpisodeMetadata:
    map_id: str | None
    spawn_configuration: Mapping[str, Any]
    environment_seed: int | None

    def spawn_id(self, agent_id):
        return (self.spawn_configuration.get("spawn_ids", {}).get(agent_id)
                or self.spawn_configuration.get("spawn_id"))


@dataclass(frozen=True)
class EpisodeLimits:
    max_steps: int
    target_laps: int | None
    termination_mode: str
    finish_agents: tuple[str, ...]
    finish_on_laps: bool

    def is_bounded(self, physical_agents, policy_agents, *, completion="race"):
        if self.max_steps > 0:
            return True
        relevant = (policy_agents if self.termination_mode == "all_trainable"
                    or completion == "policy" else physical_agents)
        return (bool(self.finish_agents) if self.termination_mode == "any_agent"
                else bool(relevant) and set(relevant) <= set(self.finish_agents))
