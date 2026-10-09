"""Collision, timeout, and opponent-outcome rewards."""
from __future__ import annotations

import math
from typing import Dict

from wrappers.rewards.base import RewardComponent


class CollisionRewardComponent(RewardComponent):
    """One-time penalty when the ego agent collides this step."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -200.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        collided = info.get("terminal_reason") == "collision"
        if collided:
            return {"collision/penalty": self.penalty}
        return {}


class TimeoutPenaltyComponent(RewardComponent):
    """Sparse penalty when an episode ends by time limit without crash outcome."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -100.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        if info.get("terminal_reason") != "time_limit":
            return {}
        return {"timeout/penalty": self.penalty}


class OpponentCrashBonusComponent(RewardComponent):
    """Local reward for each opposing car's collision terminal, once per race.

    Uses all opponent IDs from the trainer, not just the configured target.
    This rewards an outcome without attributing who caused the collision.
    Simultaneous ego/opponent crashes count; finishes and timeouts do not.
    """

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 1.0))
        if not math.isfinite(self.bonus) or self.bonus < 0:
            raise ValueError("opponent_crash_bonus.bonus must be finite and nonnegative")
        self.reset()

    def reset(self) -> None:
        self._awarded: set[str] = set()

    def compute(self, step_info: dict) -> Dict[str, float]:
        infos = step_info.get("all_infos") or {}
        opponents = set(step_info.get("opponent_agent_ids") or ())
        newly_crashed = {
            aid for aid in opponents - self._awarded
            if (infos.get(aid) or {}).get("terminal_reason") == "collision"
        }
        self._awarded.update(newly_crashed)
        if not newly_crashed:
            return {}
        return {"opponent_crash/bonus": self.bonus * len(newly_crashed)}
