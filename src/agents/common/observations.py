"""Per-learner observations with padding confined to batched policy/storage I/O."""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np


def observation_layout(contract):
    """Return the driving prefix contract, prefix width, and full input width."""
    if not isinstance(contract, dict):
        raise ValueError("Actor transfer requires explicit observation contracts")
    common = deepcopy(contract)
    obs = common["observation"]
    unknown = set(obs) - {"lidar", "frenet_vehicle_track", "frenet_neighbors"}
    if unknown or not all(obs.get(k, {}).get("enabled") for k in ("lidar", "frenet_vehicle_track")):
        raise ValueError("Actor transfer requires the LiDAR/Frenet driving layout")
    driving = int(common["lidar_beams"]) + 10 + 2 * int(obs["frenet_vehicle_track"].get("points", 20))
    neighbors = obs.pop("frenet_neighbors", {})
    extra = 0
    if neighbors.get("enabled"):
        ids = len(neighbors.get("agent_ids") or [])
        extra = int(neighbors.get("max_neighbors", 1)) * (5 + int(neighbors.get("include_team", False)) + ids) + ids
    return common, driving, driving + extra


def pack_observations(agent_ids, observations, obs_dims, width):
    if not isinstance(observations, Mapping) and len(observations) != len(agent_ids):
        raise ValueError("Observation batch shape must match the agent IDs")
    rows = np.zeros((len(agent_ids), width), dtype=np.float32)
    for i, aid in enumerate(agent_ids):
        value = observations[aid] if isinstance(observations, Mapping) else observations[i]
        value = np.asarray(value, dtype=np.float32)
        size = obs_dims[aid]
        if value.shape == (width,) and size < width:
            if np.any(value[size:] != 0):
                raise ValueError(f"Nonzero observation padding for {aid}")
        elif value.shape != (size,):
            raise ValueError(f"Observation for {aid} must have shape ({size},), got {value.shape}")
        rows[i, :size] = value[:size]
    return rows
