
from typing import Optional
from contextlib import nullcontext

import numpy as np
from numba import njit

from physics.integration import Integrator
from physics.vehicle import RaceCar
from physics.collision_models import (
    collision_multiple, swept_contact, swept_map_contact, map_motion_trace,
)


ENV_COLLISION_IDX = -2


def _body_velocity_from_state(state: np.ndarray) -> tuple[float, float]:
    """Return longitudinal/lateral velocity from ``[speed, slip_angle]``.

    The single-track model stores speed magnitude at ``state[3]`` and body
    slip angle at ``state[6]``.  Downstream observations expose body-frame
    components, so both axes must include the corresponding trigonometric
    projection.
    """
    speed = float(state[3])
    # vehicle_dynamics_st switches to its no-slip kinematic equations below
    # this threshold; state[6] can still retain an older dynamic-model value.
    if abs(speed) < 0.5:
        return speed, 0.0
    slip_angle = float(state[6])
    return speed * float(np.cos(slip_angle)), speed * float(np.sin(slip_angle))


@njit(cache=True)
def _merge_collision_results(
    agent_flags: np.ndarray,
    agent_indices: np.ndarray,
    env_mask: np.ndarray,
    out_collisions: np.ndarray,
    out_indices: np.ndarray,
    env_only_idx: int,
):
    count = out_collisions.shape[0]
    for i in range(count):
        env_hit = env_mask[i]
        agent_hit = agent_flags[i] != 0.0
        if agent_hit:
            out_collisions[i] = 1.0
            out_indices[i] = int(agent_indices[i])
        elif env_hit:
            out_collisions[i] = 1.0
            out_indices[i] = env_only_idx
        else:
            out_collisions[i] = 0.0
            out_indices[i] = -1


class Simulator(object):


    def __init__(
        self,
        params,
        num_agents,
        seed,
        time_step=0.01,
        *,
        integrator="RK4",
        lidar_dist=0.0,
        num_beams=1080,
        wall_collision_response=True,
    ):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step

            lidar_dist (float, default=0): forward LiDAR offset from the vehicle state reference

        Returns:
            None
        """
        self.num_agents = num_agents
        self.profile_region = None
        self.profile_counters = None
        self.wall_collision_response = bool(wall_collision_response)
        self.seed = seed
        self.time_step = time_step
        self.num_beams = int(num_beams)
        if isinstance(integrator, Integrator):
            integrator = integrator.value
        self.integrator = str(integrator)

        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3), dtype=np.float32)
        self.agents = []
        self.collisions = np.zeros((self.num_agents,), dtype=np.float32)
        self.collidable_mask = np.ones(self.num_agents, dtype=np.bool_)
        self.collision_idx = -1 * np.ones((self.num_agents,), dtype=np.int32)
        self.collision_times = np.full(self.num_agents, np.inf)
        

        # Cars within one environment share a scanner/distance transform. Each
        # Simulator owns a distinct scanner so an evaluation map cannot replace
        # the concurrently active training map.
        shared_scan_simulator = None
        for i in range(self.num_agents):
            agent = RaceCar(
                params,
                self.seed,
                time_step=self.time_step,
                num_beams=self.num_beams,
                integrator=self.integrator,
                lidar_dist=lidar_dist,
                scan_simulator=shared_scan_simulator,
            )
            if shared_scan_simulator is None:
                shared_scan_simulator = agent.scan_simulator
            self.agents.append(agent)
        self.scan_simulator = shared_scan_simulator

        self._state_dim = self.agents[0].physics_state.shape[0] if self.agents else 0
        self._state_buffer = np.zeros((self.num_agents, self._state_dim), dtype=np.float64)
        self._pose_buffer = np.zeros((self.num_agents, 3), dtype=np.float64)
        self._verts_buffer_f64 = np.zeros((self.num_agents, 4, 2), dtype=np.float64)
        self._verts_buffer_f32 = np.zeros((self.num_agents, 4, 2), dtype=np.float32)
        self._scan_buffer = np.zeros((self.num_agents, self.num_beams), dtype=np.float32)
        self._env_collision_mask = np.zeros((self.num_agents,), dtype=np.bool_)
        opp_count = max(self.num_agents - 1, 1)
        self._opp_pose_buffer = np.zeros((opp_count, 3), dtype=np.float32)
        self._linear_vels_x = np.zeros((self.num_agents,), dtype=np.float32)
        self._linear_vels_y = np.zeros((self.num_agents,), dtype=np.float32)
        self._ang_vels_z = np.zeros((self.num_agents,), dtype=np.float32)



    def set_map(self, map_path, map_ext, *, map_data=None):
        """
        Sets the map of the environment and sets the map for scan simulator of each agent

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        configured = set()
        for agent in self.agents:
            if id(agent.scan_simulator) not in configured:
                agent.set_map(map_path, map_ext, map_data=map_data)
                configured.add(id(agent.scan_simulator))

    def _ensure_scan_capacity(self, beam_count: int) -> None:
        if beam_count <= 0:
            beam_count = 1
        if self._scan_buffer.shape[1] == beam_count:
            return
        self._scan_buffer = np.zeros((self.num_agents, beam_count), dtype=np.float32)


    def update_params(self, params, agent_idx=-1):
        """
        Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def reseed(self, seed):
        """
        Update the simulator RNG seed and propagate it to all agents.

        Args:
            seed (int): New seed value.
        """
        self.seed = int(seed)
        for agent in self.agents:
            if hasattr(agent, "set_seed"):
                agent.set_seed(self.seed)

    def _resolve_swept_contacts(self, env_mask):
        """Resolve events chronologically, rechecking motion against stopped cars.

        Physical wall contacts stop even when predictive TTC response is disabled.
        The flag wall_collision_response still controls predictive TTC stops only.
        Equal-time pair events use the lowest other-agent index; a pair takes
        precedence over a wall at the same time, matching the public index API.
        """
        counters = self.profile_counters
        visible = np.flatnonzero(self.collidable_mask)
        count = self.num_agents
        stopped = np.full(count, np.inf)
        wall_times = np.full(count, np.inf)
        col_flags = np.zeros(count)
        hit_indices = np.full(count, -1, dtype=np.int64)
        extents = [np.array([a.geometry.front, a.geometry.rear,
                             a.geometry.left, a.geometry.right]) for a in self.agents]
        predictive = [bool(a.in_collision) for a in self.agents]
        for index in visible:
            agent = self.agents[index]
            scanner = agent.scan_simulator
            if getattr(scanner, "map_height", None) is not None:
                with self.profile_region("trajectory_map_transform") if self.profile_region else nullcontext():
                    local_trace = map_motion_trace(agent._pose_trace, scanner.origin)
                with self.profile_region("occupancy_sweep") if self.profile_region else nullcontext():
                    wall_times[index] = swept_map_contact(
                        local_trace, extents[index], scanner.map_img, scanner.map_resolution, counters)
        now = 0.
        tie_tolerance = 1e-8  # step fraction, only for deterministic grouping
        # Every event group stops at least one previously unstopped car.
        for group in range(count + 1):
            if counters is not None:
                counters[14] += 1  # event search passes, including final miss
            events = []
            for index in visible:
                if not np.isfinite(stopped[index]) and np.isfinite(wall_times[index]):
                    events.append((wall_times[index], int(index), ENV_COLLISION_IDX))
            for position, index in enumerate(visible):
                for other in visible[position + 1:]:
                    if np.isfinite(stopped[index]) and np.isfinite(stopped[other]):
                        continue
                    if counters is not None:
                        counters[15] += 1
                        if group > 0:
                            counters[16] += 1  # repeated pair query after an event
                    with self.profile_region("pair_sweep") if self.profile_region else nullcontext():
                        time = swept_contact(self.agents[index]._pose_trace, extents[index],
                                             self.agents[other]._pose_trace, extents[other],
                                             start=now, stop_a=stopped[index], stop_b=stopped[other],
                                             counters=counters)
                    if np.isfinite(time):
                        events.append((time, int(index), int(other)))
            if not events:
                break
            now = min(event[0] for event in events)
            for time, index, other in sorted(events):
                if time > now + tie_tolerance:
                    continue
                participants = ((index, other),) if other == ENV_COLLISION_IDX else ((index, other), (other, index))
                for participant, obstacle in participants:
                    stopped[participant] = min(stopped[participant], now)
                    first = self.collision_times[participant] / self.time_step
                    if not np.isfinite(first) or now < first - tie_tolerance:
                        self.collision_times[participant] = now * self.time_step
                        hit_indices[participant] = obstacle
                    elif abs(now - first) <= tie_tolerance and obstacle >= 0:
                        previous = hit_indices[participant]
                        if previous < 0 or obstacle < previous:
                            hit_indices[participant] = obstacle
                    if obstacle == ENV_COLLISION_IDX:
                        env_mask[participant] = True
                    else:
                        col_flags[participant] = 1.
        for index in visible:
            agent = self.agents[index]
            if np.isfinite(stopped[index]):
                if counters is not None:
                    counters[17] += 1
                with self.profile_region("contact_resolution") if self.profile_region else nullcontext():
                    agent.stop_at_contact(stopped[index])
                with self.profile_region("post_contact_rescan") if self.profile_region else nullcontext():
                    agent.compute_scan()
            elif predictive[index]:
                env_mask[index] = True
                self.collision_times[index] = self.time_step
                if self.wall_collision_response:
                    # CCD certified this endpoint free of physical contact.
                    # A predictive TTC stop now holds here rather than rewinding
                    # an entire step, which could introduce new pair overlaps.
                    if counters is not None:
                        counters[18] += 1
                    with self.profile_region("contact_resolution") if self.profile_region else nullcontext():
                        agent.stop_at_contact(1.)
                    with self.profile_region("post_contact_rescan") if self.profile_region else nullcontext():
                        agent.compute_scan()
        # Only pair indices feed the merger; environment-only contacts use -2.
        return col_flags, hit_indices

    def step(self, control_inputs: np.ndarray) -> dict:
        """
        Advance all agents one physics step and return vectorized observations.

        Args:
            control_inputs: (N,2) array of steering rad and speed reference:
                chassis m/s for legacy_st, wheel rad/s for combined_slip_st.

        Returns:
            obs_dict dict[str, np.ndarray] with shapes:
                scans:(N, num_beams), poses_x/y/theta:(N,), linear_vels_x/y:(N,), ang_vels_z:(N,), collisions:(N,)
            collisions are 0/1 for THIS step only (per-step semantics).
        """
        N = self.num_agents
        if not isinstance(control_inputs, np.ndarray) or control_inputs.shape != (N, 2):
            raise ValueError(f"control_inputs must have shape ({N}, 2)")
        if not np.isfinite(control_inputs).all():
            raise ValueError("control_inputs must be finite")
        if any(not np.isfinite(agent.physics_state).all() for agent in self.agents):
            raise ValueError("Vehicle states must be finite before stepping")

        self.collisions.fill(0.0)
        self.collision_idx.fill(-1)
        self.collision_times.fill(np.inf)
        env_collision_mask = self._env_collision_mask
        env_collision_mask.fill(False)

        self._ensure_scan_capacity(self.num_beams)

        state_buffer = self._state_buffer
        pose_buffer = self._pose_buffer
        verts_buffer_f64 = self._verts_buffer_f64
        verts_buffer_f32 = self._verts_buffer_f32

        for i, agent in enumerate(self.agents):
            np.copyto(state_buffer[i], agent.physics_state)
            agent.update_pose(float(control_inputs[i, 0]), float(control_inputs[i, 1]))

        with self.profile_region("physics_collisions") if self.profile_region else nullcontext():
            col_flags, hit_idx = self._resolve_swept_contacts(env_collision_mask)

        for i, agent in enumerate(self.agents):
            pose_buffer[i] = agent.state[[0, 1, 4]]
            with self.profile_region("physics_collisions") if self.profile_region else nullcontext():
                verts = agent.geometry.world_vertices(pose_buffer[i])
                verts_buffer_f64[i] = verts
                verts_buffer_f32[i] = verts
            self.agent_poses[i] = pose_buffer[i]
            v_long, v_lat = agent.body_velocity
            self._linear_vels_x[i] = np.float32(v_long)
            self._linear_vels_y[i] = np.float32(v_lat)
            self._ang_vels_z[i] = np.float32(agent.state[5])
            scan_row = np.asarray(agent.scan, dtype=np.float32)
            if scan_row.shape[0] != self._scan_buffer.shape[1]:
                self._ensure_scan_capacity(scan_row.shape[0])
                for j in range(i):
                    np.copyto(self._scan_buffer[j], self.agents[j].scan)
                self.num_beams = self._scan_buffer.shape[1]
            np.copyto(self._scan_buffer[i], scan_row)

        # Retain endpoint GJK as a compatibility check (e.g. initial overlaps).
        with self.profile_region("physics_collisions") if self.profile_region else nullcontext():
            visible = np.flatnonzero(self.collidable_mask)
            flags, hits = collision_multiple(verts_buffer_f64[visible])
            for local, index in enumerate(visible):
                if flags[local] and not col_flags[index]:
                    col_flags[index] = 1.
                    hit_idx[index] = visible[int(hits[local])]
                    self.collision_times[index] = min(self.collision_times[index], self.time_step)

        # --- 3) opponent occlusions via LiDAR footprints, etc.
        if N > 1:
            for i, agent in enumerate(self.agents):
                if not self.collidable_mask[i]:
                    continue
                if hasattr(agent, "update_opp_poses"):
                    count = 0
                    for j in range(N):
                        if j == i or not self.collidable_mask[j]:
                            continue
                        self._opp_pose_buffer[count, 0] = self.agent_poses[j, 0]
                        self._opp_pose_buffer[count, 1] = self.agent_poses[j, 1]
                        self._opp_pose_buffer[count, 2] = self.agent_poses[j, 2]
                        count += 1
                    agent.update_opp_poses(self._opp_pose_buffer[:count])

                if hasattr(agent, "ray_cast_agents"):
                    if len(visible) == N:
                        agent.ray_cast_agents(self._verts_buffer_f32, i)
                    else:
                        agent.ray_cast_agents(self._verts_buffer_f32[visible], int(np.searchsorted(visible, i)))
                    scan_row = np.asarray(agent.scan, dtype=np.float32)
                    if scan_row.shape[0] != self._scan_buffer.shape[1]:
                        self._ensure_scan_capacity(scan_row.shape[0])
                        for j in range(N):
                            prev = np.asarray(self.agents[j].scan, dtype=np.float32)
                            np.copyto(self._scan_buffer[j], prev)
                        self.num_beams = self._scan_buffer.shape[1]
                    np.copyto(self._scan_buffer[i], scan_row)

                if getattr(agent, "in_collision", False):
                    env_collision_mask[i] = True

        with self.profile_region("physics_collisions") if self.profile_region else nullcontext():
            _merge_collision_results(
                np.asarray(col_flags, dtype=np.float64),
                np.asarray(hit_idx, dtype=np.float64),
                env_collision_mask,
                self.collisions,
                self.collision_idx,
                ENV_COLLISION_IDX,
            )
            self.collisions[~self.collidable_mask] = 0.0
            self.collision_idx[~self.collidable_mask] = -1

        obs_dict = {
            "scans": self._scan_buffer.copy(),
            "poses_x": self.agent_poses[:, 0].copy(),
            "poses_y": self.agent_poses[:, 1].copy(),
            "poses_theta": self.agent_poses[:, 2].copy(),
            "linear_vels_x": self._linear_vels_x.copy(),
            "linear_vels_y": self._linear_vels_y.copy(),
            "ang_vels_z": self._ang_vels_z.copy(),
            "collisions": self.collisions.copy(),
        }
        return obs_dict

    def freeze_agent(self, agent_idx: int) -> None:
        """Freeze a terminal car and immediately synchronize published velocities."""
        agent = self.agents[agent_idx]
        agent.freeze_motion()
        self._linear_vels_x[agent_idx], self._linear_vels_y[agent_idx] = agent.body_velocity
        self._ang_vels_z[agent_idx] = agent.state[5]

    def set_agent_speed(self, agent_idx: int, speed: float) -> None:
        """Override the longitudinal speed of a specific agent."""
        if agent_idx < 0 or agent_idx >= self.num_agents:
            return
        try:
            value = float(speed)
        except (TypeError, ValueError):
            value = 0.0

        agent = self.agents[agent_idx]
        if hasattr(agent, "set_longitudinal_speed"):
            agent.set_longitudinal_speed(value)
        else:
            agent.state[3] = value

        v_long, v_lat = (agent.body_velocity if getattr(agent, "nonlinear", False)
                                  else _body_velocity_from_state(agent.state))
        self._linear_vels_x[agent_idx] = np.float32(v_long)
        self._linear_vels_y[agent_idx] = np.float32(v_lat)
        self._ang_vels_z[agent_idx] = np.float32(agent.state[5])

    def current_observation(self, *, include_scans: bool = True) -> dict:
        """Copy the joint buffers, optionally omitting LiDAR for state-only users."""
        return {
            **({"scans": self._scan_buffer.copy()} if include_scans else {}),
            "poses_x": self.agent_poses[:, 0].copy(),
            "poses_y": self.agent_poses[:, 1].copy(),
            "poses_theta": self.agent_poses[:, 2].copy(),
            "linear_vels_x": self._linear_vels_x.copy(),
            "linear_vels_y": self._linear_vels_y.copy(),
            "ang_vels_z": self._ang_vels_z.copy(),
            "collisions": self.collisions.copy(),
        }

    def refresh_scans(self):
        """Refresh sensing after removal without advancing physics or collision events."""
        visible = np.flatnonzero(self.collidable_mask)
        for local, index in enumerate(visible):
            agent = self.agents[index]
            agent.compute_scan()
            others = visible[visible != index]
            if hasattr(agent, "update_opp_poses"):
                agent.update_opp_poses(self.agent_poses[others])
            if hasattr(agent, "ray_cast_agents"):
                agent.ray_cast_agents(self._verts_buffer_f32[visible], local)
            np.copyto(self._scan_buffer[index], agent.scan)


    def reset(self, poses: np.ndarray, velocities: Optional[np.ndarray] = None, *, friction_mu=None, agent_indices=None) -> dict:
        """
        Reset the simulator to the given poses and return initial vectorized observations.

        Args:
            poses: np.ndarray of shape (N, 3) with (x, y, theta) per agent.
            velocities: Optional np.ndarray of shape (N,) with initial forward speeds (m/s) per agent.
            agent_indices: Optional subset to reset; other vehicle states and velocities are preserved.

        Returns:
            obs_dict with scans, poses, velocities, collisions.
        """
        if not isinstance(poses, np.ndarray) or poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"poses must be an array with shape (N,3); got {getattr(poses, 'shape', None)}")
        if poses.shape[0] != self.num_agents:
            raise ValueError("Number of poses for reset does not match number of agents.")
        if not np.isfinite(poses).all():
            raise ValueError("Reset poses must be finite")
        if velocities is not None:
            if not isinstance(velocities, np.ndarray):
                velocities = np.array(velocities, dtype=np.float32)
            if velocities.ndim != 1 or velocities.shape[0] != self.num_agents:
                raise ValueError(f"velocities must be a 1D array with {self.num_agents} elements; got shape {velocities.shape}")
            if np.isinf(velocities).any():
                raise ValueError("Reset velocities must be finite (NaN means use the default)")
        if agent_indices is None:
            self.collidable_mask.fill(True)
        else:
            self.collidable_mask[list(agent_indices)] = True

        N = self.num_agents

        self.collisions.fill(0.0)
        self.collision_idx.fill(-1)
        self.collision_times.fill(np.inf)
        env_collision_mask = self._env_collision_mask
        env_collision_mask.fill(False)

        self._ensure_scan_capacity(self.num_beams)

        pose_buffer = self._pose_buffer

        selected = set(range(N) if agent_indices is None else agent_indices)
        for i, agent in enumerate(self.agents):
            if i in selected and friction_mu is not None:
                if not agent.nonlinear:
                    raise ValueError("Episode friction requires combined_slip_st")
                agent.reset(poses[i], friction_mu=friction_mu)
            elif i in selected:
                agent.reset(poses[i])

            # Set custom initial velocity if provided (skip if NaN)
            if i in selected and velocities is not None and i < len(velocities):
                vel = float(velocities[i])
                if not np.isnan(vel):
                    # agent.state[3] is v_long (longitudinal velocity)
                    agent.set_longitudinal_speed(vel)

            pose_buffer[i, 0] = agent.state[0]
            pose_buffer[i, 1] = agent.state[1]
            pose_buffer[i, 2] = agent.state[4]

            self.agent_poses[i, 0] = np.float32(agent.state[0])
            self.agent_poses[i, 1] = np.float32(agent.state[1])
            self.agent_poses[i, 2] = np.float32(agent.state[4])

            v_long, v_lat = (agent.body_velocity if getattr(agent, "nonlinear", False)
                                  else _body_velocity_from_state(agent.state))
            yaw_rate = float(agent.state[5])

            self._linear_vels_x[i] = np.float32(v_long)
            self._linear_vels_y[i] = np.float32(v_lat)
            self._ang_vels_z[i] = np.float32(yaw_rate)

            scan_row = np.asarray(agent.compute_scan(), dtype=np.float32)
            if scan_row.shape[0] != self._scan_buffer.shape[1]:
                self._ensure_scan_capacity(scan_row.shape[0])
                self.num_beams = self._scan_buffer.shape[1]
            np.copyto(self._scan_buffer[i], scan_row)

        for i in range(N):
            agent = self.agents[i]
            verts = agent.geometry.world_vertices(pose_buffer[i])
            self._verts_buffer_f64[i, :, :] = verts
            self._verts_buffer_f32[i, :, :] = verts
        visible = np.flatnonzero(self.collidable_mask)
        flags, hits = collision_multiple(self._verts_buffer_f64[visible])
        col_flags = np.zeros(N, dtype=np.float64)
        hit_idx = np.full(N, -1, dtype=np.int64)
        col_flags[visible] = flags
        for local, hit in enumerate(hits):
            if hit >= 0:
                hit_idx[visible[local]] = visible[int(hit)]

        if N > 1:
            for i, agent in enumerate(self.agents):
                if not self.collidable_mask[i]:
                    continue
                if hasattr(agent, "update_opp_poses"):
                    count = 0
                    for j in range(N):
                        if j == i or not self.collidable_mask[j]:
                            continue
                        self._opp_pose_buffer[count, 0] = self.agent_poses[j, 0]
                        self._opp_pose_buffer[count, 1] = self.agent_poses[j, 1]
                        self._opp_pose_buffer[count, 2] = self.agent_poses[j, 2]
                        count += 1
                    agent.update_opp_poses(self._opp_pose_buffer[:count])

                if hasattr(agent, "ray_cast_agents"):
                    agent.ray_cast_agents(self._verts_buffer_f32[visible], int(np.searchsorted(visible, i)))
                    scan_row = np.asarray(agent.scan, dtype=np.float32)
                    if scan_row.shape[0] != self._scan_buffer.shape[1]:
                        self._ensure_scan_capacity(scan_row.shape[0])
                        self.num_beams = self._scan_buffer.shape[1]
                    np.copyto(self._scan_buffer[i], scan_row)

        _merge_collision_results(
            np.asarray(col_flags, dtype=np.float64),
            np.asarray(hit_idx, dtype=np.float64),
            env_collision_mask,
            self.collisions,
            self.collision_idx,
            ENV_COLLISION_IDX,
        )
        self.collision_times[self.collisions != 0.] = 0.

        obs_dict = {
            "scans": self._scan_buffer.copy(),
            "poses_x": self.agent_poses[:, 0].copy(),
            "poses_y": self.agent_poses[:, 1].copy(),
            "poses_theta": self.agent_poses[:, 2].copy(),
            "linear_vels_x": self._linear_vels_x.copy(),
            "linear_vels_y": self._linear_vels_y.copy(),
            "ang_vels_z": self._ang_vels_z.copy(),
            "collisions": self.collisions.copy(),
        }
        return obs_dict
