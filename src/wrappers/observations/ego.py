"""LiDAR observations."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class LidarComponent(ObservationComponent):
    """Raw lidar scan — optionally normalized by lidar_range.

    The physical scanner range is independent of the policy normalization
    distance. Invalid inputs become zero; valid raw ranges retain their units.
    """

    def __init__(self, n_beams: int, lidar_range: float, normalize: bool = True) -> None:
        if not np.isfinite(lidar_range) or lidar_range <= 0:
            raise ValueError("LiDAR normalization distance must be finite and positive")
        self._n_beams = n_beams
        self._normalize = normalize
        # Store as float32 scalar so multiply stays in float32 arithmetic.
        self._inv_range = np.float32(1.0 / float(lidar_range)) if normalize else None

    @property
    def dim(self) -> int:
        return self._n_beams

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        """Write normalized lidar scan into *out* (zero allocation on fast path)."""
        scan_raw = raw_obs.get("lidar")
        if scan_raw is None:
            out.fill(0.0)
            return

        scan = np.asarray(scan_raw, dtype=np.float32)
        # Resize only when shape doesn't match (rare — env normally provides correct shape).
        if scan.shape[0] != self._n_beams:
            scan = np.resize(scan, (self._n_beams,))

        if self._normalize:
            np.multiply(scan, self._inv_range, out=out)
        else:
            np.copyto(out, scan)
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self._normalize:
            np.clip(out, 0.0, 1.0, out=out)
        else:
            np.maximum(out, 0.0, out=out)

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(self._n_beams, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out
