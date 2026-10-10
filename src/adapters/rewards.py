"""Translate task rewards into the signal exposed by an RL adapter."""
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RewardMapping:
    mode: str = "individual"
    reduction: str = "mean"

    def __post_init__(self):
        if self.mode not in {"individual", "team_shared"}:
            raise ValueError("reward_mode must be individual or team_shared")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("team_reward_reduction must be mean or sum")

    def map(self, individual: Mapping[str, float], team_agents: Sequence[str],
            shared_bonus: float = 0.0) -> dict[str, float]:
        if not team_agents or not set(individual) <= set(team_agents):
            raise ValueError("Rewards must belong to the configured policy team")
        if self.mode == "individual":
            return dict(individual)
        reward = float(sum(individual.values()))
        if self.reduction == "mean":
            # Retirement does not change the learning signal's scale.
            reward /= len(team_agents)
        reward += float(shared_bonus)
        return dict.fromkeys(individual, reward)

    def from_step(self, step, team_agents):
        return self.map({aid: decision.individual_reward
                         for aid, decision in step.decisions.items()},
                        team_agents, step.team_reward)
