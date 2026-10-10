"""Lifecycle fields for training transition records."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from env.types import GlobalState


def transition_lifecycle_fields(
    info: Dict[str, Any],
    *,
    global_state: Optional[GlobalState] = None,
) -> Dict[str, Any]:
    """Build dataset lifecycle fields from detached post-decision facts."""
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
