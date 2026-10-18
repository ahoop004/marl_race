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


def observation_segments(contract, input_dim):
    """Derive ordered, half-open encoder segments from the composer contract.

    Dimension arithmetic stays in observation_layout, shared with transfer.
    CNNs require explicit contracts rather than guessing a scan from its width.
    """
    if (not isinstance(contract, dict) or type(contract.get("version")) is not int
            or contract["version"] != 1):
        raise ValueError("CNN encoder requires an explicit version 1 LiDAR/Frenet observation contract; global-state inputs are unsupported")
    obs = contract.get("observation")
    if not isinstance(obs, dict) or any(not isinstance(value, dict) for value in obs.values()):
        raise ValueError("Observation contract components must be mappings")
    for key, settings in obs.items():
        if type(settings.get("enabled", False)) is not bool:
            raise ValueError(f"Observation contract {key}.enabled must be a boolean")
    for label, value in (("lidar_beams", contract.get("lidar_beams")),
                         ("points", obs.get("frenet_vehicle_track", {}).get("points", 20)),
                         ("input_dim", input_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Observation layout {label} must be a positive integer")
    neighbors = obs.get("frenet_neighbors", {})
    if neighbors.get("enabled"):
        count = neighbors.get("max_neighbors", 1)
        ids = neighbors.get("agent_ids")
        if type(count) is not int or count <= 0 or type(neighbors.get("include_team", False)) is not bool:
            raise ValueError("Observation layout requires positive max_neighbors and boolean include_team")
        if ids is not None and (not isinstance(ids, (list, tuple)) or not ids
                or any(not isinstance(aid, str) or not aid for aid in ids) or len(set(ids)) != len(ids)):
            raise ValueError("Observation layout requires unique nonempty neighbor agent_ids")
    _, driving, total = observation_layout(contract)
    if total != input_dim:
        raise ValueError(f"Observation segments sum to {total}, but actual observation dimension is {input_dim}")
    return {"lidar": [0, contract["lidar_beams"]],
            "vehicle_track": [contract["lidar_beams"], driving], "neighbors": [driving, total]}


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
