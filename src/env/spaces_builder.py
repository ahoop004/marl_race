"""Build raw environment action and observation space specs."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from env.spaces import DictSpaceSpec, SpaceSpec


def build_action_spaces(
    possible_agents: Sequence[str],
    vehicle_params: Mapping[str, float],
) -> Tuple[SpaceSpec, Dict[str, SpaceSpec]]:
    if vehicle_params.get("model") == "combined_slip_st":
        actuators = vehicle_params["wheel_actuators"]
        low = [actuators["steering_min"], actuators["wheel_speed_min"]]
        high = [actuators["steering_max"], actuators["wheel_speed_max"]]
    else:
        low = [vehicle_params["s_min"], vehicle_params["v_min"]]
        high = [vehicle_params["s_max"], vehicle_params["v_max"]]
    single_action_space = SpaceSpec(
        shape=(2,),
        low=np.array(low, dtype=np.float32),
        high=np.array(high, dtype=np.float32),
    )
    return single_action_space, {aid: single_action_space for aid in possible_agents}


def build_observation_spaces(
    *,
    possible_agents: Sequence[str],
    agent_sensor_spec: Mapping[str, Tuple[str, ...]],
    default_sensors: Tuple[str, ...],
    central_state_dim: int,
    lidar_beam_count: int,
    lidar_range: float,
    vehicle_params: Mapping[str, float],
    target_laps: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    continuous_laps: bool = False,
) -> Dict[str, DictSpaceSpec]:
    nonlinear = vehicle_params.get("model") == "combined_slip_st"
    control_params = vehicle_params["wheel_actuators"] if nonlinear else vehicle_params
    steering_low = float(control_params["steering_min" if nonlinear else "s_min"])
    steering_high = float(control_params["steering_max" if nonlinear else "s_max"])
    speed_low = float(control_params["wheel_speed_min" if nonlinear else "v_min"])
    speed_high = float(control_params["wheel_speed_max" if nonlinear else "v_max"])

    pose_low = np.array([x_min, y_min, -np.pi], dtype=np.float32)
    pose_high = np.array([x_max, y_max, np.pi], dtype=np.float32)
    v_min = float(vehicle_params.get("v_min", -5.0))
    v_max = float(vehicle_params.get("v_max", 20.0))
    if nonlinear:
        v_min, v_max = -np.inf, np.inf
    vel_low = np.array([v_min, v_min], dtype=np.float32)
    vel_high = np.array([v_max, v_max], dtype=np.float32)

    accel_cap = float(vehicle_params.get("a_max", 10.0))
    accel_low = np.array([-accel_cap, -accel_cap], dtype=np.float32)
    accel_high = np.array([accel_cap, accel_cap], dtype=np.float32)

    ang_cap = float(vehicle_params.get("ang_vel_max", 10.0))
    if nonlinear:
        accel_low.fill(-np.inf)
        accel_high.fill(np.inf)
        ang_cap = np.inf

    lap_cap = np.inf if continuous_laps else float(target_laps)
    lap_low = np.array([0.0, 0.0], dtype=np.float32)
    lap_high = np.array([lap_cap, 1e5], dtype=np.float32)

    obs_spaces: Dict[str, DictSpaceSpec] = {}
    for aid in possible_agents:
        sensors = agent_sensor_spec.get(aid, default_sensors)
        components: Dict[str, SpaceSpec] = {}

        if "lidar" in sensors:
            # lidar_range is the policy normalization scale. Raw scans use
            # the scanner's range and additive Gaussian noise, so neither
            # that scale nor zero is a hard bound on the returned readings.
            components["lidar"] = SpaceSpec(
                shape=(lidar_beam_count,),
                low=-np.inf,
                high=np.inf,
            )
            # Raw observations retain this alias for controller/render consumers.
            components["scans"] = components["lidar"]
        if "pose" in sensors:
            components["pose"] = SpaceSpec(shape=(3,), low=pose_low, high=pose_high)
        if "velocity" in sensors:
            components["velocity"] = SpaceSpec(shape=(2,), low=vel_low, high=vel_high)
        if "acceleration" in sensors:
            components["acceleration"] = SpaceSpec(shape=(2,), low=accel_low, high=accel_high)
        if "angular_velocity" in sensors:
            components["angular_velocity"] = SpaceSpec(
                shape=(), low=-ang_cap, high=ang_cap,
            )
        if "target_pose" in sensors:
            # Agents without a configured target return the zero pose, even
            # when the map's coordinates do not include the origin.
            components["target_pose"] = SpaceSpec(
                shape=(3,), low=np.minimum(pose_low, 0.0), high=np.maximum(pose_high, 0.0),
            )
        if "target_collision" in sensors:
            components["target_collision"] = SpaceSpec(
                shape=(), low=0.0, high=1.0,
            )
        if "lap" in sensors:
            components["lap"] = SpaceSpec(shape=(2,), low=lap_low, high=lap_high)
        if "collision" in sensors:
            components["collision"] = SpaceSpec(
                shape=(), low=0.0, high=1.0,
            )

        # RaceEnv always attaches physical state and command references,
        # independently of the optional sensor selection. Scalars keep their
        # existing float32 representation for controllers and composers.
        components["steering_angle"] = SpaceSpec(
            shape=(), low=steering_low, high=steering_high,
        )
        components["steering_reference"] = SpaceSpec(
            shape=(), low=steering_low, high=steering_high,
        )
        speed_key = "wheel_speed" if nonlinear else "speed"
        if nonlinear:
            components["wheel_speed"] = SpaceSpec(
                shape=(), low=speed_low, high=speed_high,
            )
        components[f"{speed_key}_reference"] = SpaceSpec(
            shape=(), low=speed_low, high=speed_high,
        )
        # These are changes in the commanded reference per decision, rather
        # than the rate-limited actuator's physical acceleration.
        components[f"{speed_key}_reference_rate"] = SpaceSpec(
            shape=(), low=-np.inf, high=np.inf,
        )

        components["state"] = SpaceSpec(
            shape=(central_state_dim,),
            low=np.full(central_state_dim, -np.inf, dtype=np.float32),
            high=np.full(central_state_dim, np.inf, dtype=np.float32),
        )
        obs_spaces[aid] = DictSpaceSpec(spaces=components)

    return obs_spaces
