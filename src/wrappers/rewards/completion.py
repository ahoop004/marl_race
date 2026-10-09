"""Individual and team progress, lap completion, and race-duration rewards."""
from __future__ import annotations

from typing import Dict, Optional
import math

from wrappers.rewards.base import RewardComponent
from metrics.racing_eval import team_finish_result


def _centerline_info(step_info: dict) -> dict:
    info = step_info.get("info") or {}
    centerline = info.get("centerline", {}) if isinstance(info, dict) else {}
    return centerline if isinstance(centerline, dict) else {}


def _float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ProgressDeltaBonusComponent(RewardComponent):
    """Signed progress with optional exclusive boundary/collision costs."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 100.0))
        self.positive_only = bool(config.get("positive_only", True))
        self.units = str(config.get("units", "lap_fraction"))
        if self.units not in {"lap_fraction", "metres"}:
            raise ValueError("Progress units must be lap_fraction or metres")
        self.boundary_penalty = config.get("boundary_penalty")
        self.collision_penalty = config.get("collision_penalty")
        max_delta = config.get("max_delta")
        self.max_delta = _float_or_none(max_delta) if max_delta is not None else None

    def compute(self, step_info: dict) -> Dict[str, float]:
        if (self.collision_penalty is not None and
                (step_info.get("info") or {}).get("terminal_reason") == "collision"):
            return {"progress_delta/collision": float(self.collision_penalty)}
        if self.boundary_penalty is not None:
            boundary = (step_info.get("info") or {}).get("track_limits")
            if boundary is None:
                raise ValueError("Boundary progress reward requires track_limits facts")
            if boundary["exceeded"] or (step_info.get("info") or {}).get("boundary_event", False):
                return {"progress_delta/boundary": float(self.boundary_penalty)}
        centerline = _centerline_info(step_info)
        progress_delta = _float_or_none(centerline.get("progress_delta"))
        if progress_delta is None:
            return {}

        if self.units == "metres":
            length = _float_or_none(step_info.get("track_length"))
            if length is None or length <= 0:
                raise ValueError("Metre progress reward requires a positive track_length")
            progress_delta *= length

        if self.positive_only:
            progress_delta = max(progress_delta, 0.0)
        if self.max_delta is not None:
            bound = abs(self.max_delta)
            if self.positive_only:
                progress_delta = min(progress_delta, bound)
            else:
                progress_delta = max(-bound, min(bound, progress_delta))

        return {"progress_delta/bonus": self.weight * progress_delta}


class StepTimePenaltyComponent(RewardComponent):
    """Time cost per physics step, optionally specified per simulated second."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config["penalty"] if config.get("penalty") is not None else -0.01)
        self.per_second = config.get("per_second")
        if self.per_second is not None:
            if config.get("penalty") is not None:
                raise ValueError("Specify either step_time_penalty.penalty or per_second")
            self.per_second = float(self.per_second)
            if not math.isfinite(self.per_second) or self.per_second > 0:
                raise ValueError("Time cost per_second must be finite and nonpositive")
        self.apply_on_terminal = bool(config.get("apply_on_terminal", True))

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not self.apply_on_terminal and (
            step_info.get("done") or step_info.get("terminated") or step_info.get("truncated")
        ):
            return {}
        penalty = self.penalty
        if self.per_second is not None:
            timestep = float(step_info["timestep"])
            if not math.isfinite(timestep) or timestep <= 0:
                raise ValueError("Time cost requires a positive finite timestep")
            penalty = self.per_second * timestep
        return {"step_time/penalty": penalty}


class LapCompletionComponent(RewardComponent):
    """Sparse final-race bonus tied only to explicit completion facts."""

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 150.0))
        self.require_clean = bool(config.get("require_clean", True))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        if not bool(info.get("lap_crossed", False)):
            return {}
        if not bool(info.get("race_completed", False)):
            return {}

        if self.require_clean and bool(info.get("collision", False)):
            return {}

        return {"lap_completion/bonus": self.bonus}


class TeamRaceResultComponent(RewardComponent):
    """Shared result increments, computed once per joint step before distribution.

    Combined rank points are paid when each clean finish becomes known, and the
    both-finished bonus when the second teammate finishes. First/sweep bonuses
    are likewise paid as soon as decided; they never wait for parked opponents.
    """

    scope = "team"

    def __init__(self, config: dict) -> None:
        self.objective = str(config.get("objective", "combined"))
        if self.objective not in {"combined", "first_place", "sweep"}:
            raise ValueError("team_race_result objective must be combined, first_place, or sweep")
        self.contract = {"objective": self.objective}
        keys = ("rank_bonus", "both_finish_bonus") if self.objective == "combined" else ("win_bonus",)
        for key in keys:
            value = float(config[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"team_race_result {key} must be finite and nonnegative")
            self.contract[key] = value
        self.reset()

    def reset(self) -> None:
        self._paid: Dict[str, float] = {}

    def compute(self, step_info: dict) -> Dict[str, float]:
        result = team_finish_result(
            step_info["all_infos"], step_info["trainable_agent_ids"],
            step_info["opponent_agent_ids"],
        )
        if self.objective == "combined":
            scores = {
                "team_result/rank": self.contract["rank_bonus"] * result["rank_score"],
                "team_result/both_finished": self.contract["both_finish_bonus"] * result["both_finished"],
            }
        else:
            scores = {f"team_result/{self.objective}": self.contract["win_bonus"] * result[self.objective]}
        increments = {key: value - self._paid.get(key, 0.0) for key, value in scores.items()}
        self._paid = scores
        return {key: value for key, value in increments.items() if value != 0.0}
