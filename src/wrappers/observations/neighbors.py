"""Nearby-agent Frenet observations."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from wrappers.observations.base import ObservationComponent


_FIELDS = ("delta_s", "delta_d", "delta_vs", "delta_vd")
_DEFAULT_MAXIMA = {
    "delta_s": 20.0,
    "delta_d": 5.0,
    "delta_vs": 20.0,
    "delta_vd": 10.0,
}


class FrenetNeighborsComponent(ObservationComponent):
    """Nearest-agent slots ``[Δs, Δd, Δvs, Δvd, present]``.

    Neighbors are ordered by absolute wrapped longitudinal distance, with
    agent ID used only as a deterministic tie-breaker. Missing slots are zero.
    With ``include_team``, each slot appends an explicit ``is_teammate`` flag.
    With ``agent_ids``, each slot then appends a one-hot vehicle ID in that
    configured order, followed by one ego-ID one-hot after all slots. IDs stay
    stable when neighbors reorder or disappear; empty slots remain all zero.
    """

    def __init__(
        self,
        *,
        max_neighbors: int,
        maxima: Mapping[str, float] | None = None,
        clip: bool = False,
        include_team: bool = False,
        agent_ids: Sequence[str] | None = None,
    ) -> None:
        self.max_neighbors = max(int(max_neighbors), 1)
        configured = dict(_DEFAULT_MAXIMA)
        configured.update(dict(maxima or {}))
        self._maxima = np.asarray(
            [max(abs(float(configured[field])), 1e-6) for field in _FIELDS],
            dtype=np.float32,
        )
        self.clip = bool(clip)
        self.include_team = bool(include_team)
        if agent_ids is not None and (
            not isinstance(agent_ids, (list, tuple)) or not agent_ids
            or any(not isinstance(aid, str) or not aid for aid in agent_ids)
            or len(set(agent_ids)) != len(agent_ids)
        ):
            raise ValueError("Neighbor agent_ids must be a nonempty list of unique vehicle IDs")
        self.agent_ids = tuple(agent_ids or ())
        self._agent_index = {aid: index for index, aid in enumerate(self.agent_ids)}
        self.slot_dim = 5 + int(self.include_team) + len(self.agent_ids)

    @property
    def dim(self) -> int:
        return self.slot_dim * self.max_neighbors + len(self.agent_ids)

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        out.fill(0.0)
        if self.agent_ids:
            ego_id = info.get("agent_id")
            if ego_id not in self._agent_index:
                raise ValueError(f"Neighbor identity requires a configured ego agent_id, got {ego_id!r}")
            out[self.slot_dim * self.max_neighbors + self._agent_index[ego_id]] = 1.0
        neighbors = info.get("frenet_neighbors", []) if isinstance(info, dict) else []
        if not isinstance(neighbors, (list, tuple)):
            return
        for slot, neighbor in enumerate(neighbors[: self.max_neighbors]):
            if not isinstance(neighbor, Mapping):
                continue
            start = self.slot_dim * slot
            values = np.asarray(
                [_finite_number(neighbor.get(field)) for field in _FIELDS],
                dtype=np.float32,
            )
            out[start : start + 4] = values / self._maxima
            out[start + 4] = 1.0
            if self.include_team:
                if "is_teammate" not in neighbor:
                    raise ValueError("Team neighbor observations require explicit agent_teams facts")
                out[start + 5] = float(neighbor["is_teammate"])
            if self.agent_ids:
                neighbor_id = neighbor.get("agent_id")
                if neighbor_id not in self._agent_index or neighbor_id == ego_id:
                    raise ValueError(f"Invalid neighbor agent_id: {neighbor_id!r}")
                out[start + 5 + int(self.include_team) + self._agent_index[neighbor_id]] = 1.0
        np.nan_to_num(out, copy=False)
        if self.clip:
            np.clip(out, -1.0, 1.0, out=out)


def _finite_number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


__all__ = ["FrenetNeighborsComponent"]
