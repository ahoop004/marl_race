"""Shared reward-context assembly for training loops."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from env.types import GlobalState
from tasks.reward_context import build_reward_context


def validate_team_reward_composers(composers: Dict, *, trainable_ids: Sequence[str],
                                   opponent_ids: Sequence[str], reward_mode: str,
                                   critic_mode: str, team_return_mode: str,
                                   action_repeat: int) -> bool:
    """Reject shared reward terms unless collection can credit the whole team."""
    contracts = [getattr(composers.get(aid), "team_contract", []) for aid in trainable_ids]
    if not any(contracts):
        return False
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("All teammates must use identical team reward components")
    if (len(trainable_ids) != 2 or len(opponent_ids) != 2
            or reward_mode != "team_shared" or critic_mode != "shared_team"
            or team_return_mode != "joint" or action_repeat != 1):
        raise ValueError("Team race rewards require 2v2 MAPPO with joint team returns, shared_team critic, team_shared reward, and action_repeat=1")
    return True


def transition_lifecycle_fields(
    env: Any,
    info: Dict[str, Any],
    *,
    global_state: Optional[GlobalState] = None,
) -> Dict[str, Any]:
    """Build dataset lifecycle fields from one post-decision environment view."""
    if global_state is None:
        try:
            global_state = env.get_global_state()
        except Exception:
            global_state = None
    masks = getattr(global_state, "masks", {})
    return {
        "lap_crossed": bool(info.get("lap_crossed", False)),
        "lap_count": int(info.get("lap_count", 0)),
        "target_laps": int(info.get("target_laps", 1)),
        "race_completed": bool(info.get("race_completed", False)),
        "terminal_reason": info.get("terminal_reason"),
        "lifecycle_status": str(info.get("status", "active")),
        "finish_position": info.get("finish_position"),
        "lifecycle_masks": {
            key: np.asarray(value, dtype=bool).copy()
            for key, value in masks.items()
            if key in {
                "active_mask",
                "finished_mask",
                "crashed_mask",
                "truncated_mask",
            }
        },
    }
