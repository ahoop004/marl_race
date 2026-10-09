"""RewardComposer — assembles RewardComponents from config."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Type

from core.scenario import load_yaml_config

from wrappers.rewards.base import RewardComponent
from wrappers.rewards.events import (
    CollisionRewardComponent,
    OpponentCrashBonusComponent,
    TimeoutPenaltyComponent,
)
from wrappers.rewards.interaction import TeamSupportComponent
from wrappers.rewards.completion import (
    LapCompletionComponent,
    ProgressDeltaBonusComponent,
    StepTimePenaltyComponent,
    TeamRaceResultComponent,
)


COMPONENT_REGISTRY: Dict[str, Type[RewardComponent]] = {
    "team_race_result": TeamRaceResultComponent,
    "collision": CollisionRewardComponent,
    "progress_delta_bonus": ProgressDeltaBonusComponent,
    "step_time_penalty": StepTimePenaltyComponent,
    "lap_completion": LapCompletionComponent,
    "opponent_crash_bonus": OpponentCrashBonusComponent,
    "team_support": TeamSupportComponent,
    "timeout_penalty": TimeoutPenaltyComponent,
}


class RewardComposer:
    """Sums enabled RewardComponents into a total scalar + breakdown dict.

    Built from an inline config dict or a user-supplied YAML file.
    """

    def __init__(self, components: List[RewardComponent]) -> None:
        self._components = components

    def reset(self) -> None:
        for c in self._components:
            c.reset()

    @property
    def team_contract(self) -> List[Dict]:
        return [dict(c.contract) for c in self._components if getattr(c, "scope", None) == "team"]

    def compute(self, step_info: dict, *, team: bool = False) -> Tuple[float, Dict[str, float]]:
        """Compute local rewards by default, or shared components once per step."""
        breakdown: Dict[str, float] = {}
        for component in self._components:
            if (getattr(component, "scope", None) == "team") != team:
                continue
            breakdown.update(component.compute(step_info))
        total = sum(breakdown.values())
        return total, breakdown

    @classmethod
    def from_config(cls, reward_config: Dict) -> "RewardComposer":
        """Build from a parsed reward config dict."""
        cfg = reward_config.get("reward", reward_config)
        unknown = sorted(set(cfg) - set(COMPONENT_REGISTRY))
        if unknown:
            raise ValueError(f"RewardComposer: unknown reward component(s): {unknown}")
        components: List[RewardComponent] = []

        for key, component_cls in COMPONENT_REGISTRY.items():
            comp_cfg = cfg.get(key, {})
            if isinstance(comp_cfg, dict) and comp_cfg.get("enabled", False):
                components.append(component_cls(comp_cfg))

        if not components:
            raise ValueError("RewardComposer: no components enabled in reward config.")

        return cls(components)

    @classmethod
    def from_file(cls, path: str) -> "RewardComposer":
        """Load from a YAML reward config file path."""
        reward_config = load_yaml_config(Path(path))
        return cls.from_config(reward_config)
