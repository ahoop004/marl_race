"""Deterministic pure pursuit with LiDAR gap avoidance, in physical rad/m/s.

Geometry is cached per map. Decisions use array arithmetic and bounded path
samples; no solver, learned policy, or map distance field.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from utils.centerline import prepare_centerline_geometry, project_to_centerline


class _DecisionTiming:
    """Exclusive sequential stages, allocated only for an opt-in measured call."""
    def __init__(self):
        self.started = self.boundary = perf_counter()
        self.stages = {}
        self.stage_calls = {}
        self.context = dict(invalid_state=0, missing_scan=0, no_gap=0,
                            blocked=0, emergency_brake=0, mode_switch=0,
                            traffic_count=0, cold_start=0, map_changed=0, input_speed=0.,
                            lidar_hits=0, path_samples=0, gap_candidates=0)

    def mark(self, stage):
        now = perf_counter()
        self.stages[stage] = self.stages.get(stage, 0.) + max(0., now - self.boundary)
        self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1
        self.boundary = now


class HybridPPFTGAgent:
    def __init__(self, config):
        cfg = config.get("params", config)
        self.agent_id = cfg.get("agent_id")
        defaults = dict(max_speed=5.0, lookahead_base=1.0, lookahead_time=0.3,
                        lookahead_max=3.0, lateral_acceleration=4.0,
                        acceleration=3.0, braking=5.0, reaction_time=0.55,
                        margin=0.12, min_gap_distance=1.0, gap_fov=2.4,
                        clear_steps=5, exit_margin=0.35,
                        steering_time_constant=0.08, steering_rate=2.5,
                        curvature_spacing=0.5, path_spacing=0.2, gap_speed=3.0)
        unknown = set(cfg) - set(defaults) - {"agent_id"}
        if unknown:
            raise ValueError(f"Unknown hybrid_pp_ftg parameters: {sorted(unknown)}")
        for name, default in defaults.items():
            value = float(cfg.get(name, default))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"hybrid_pp_ftg {name} must be positive and finite")
            setattr(self, name, value)
        if (self.lookahead_max < self.lookahead_base or self.gap_fov > np.pi
                or self.clear_steps != int(self.clear_steps)):
            raise ValueError("Invalid hybrid_pp_ftg lookahead, gap FOV, or clear_steps")
        self.clear_steps = int(self.clear_steps)
        self.env = None
        self.profile_hybrid = None  # Run-owned callback; survives episode/map resets.
        self._map_source = None
        self._scan_key = None
        self.reset()

    def set_env(self, env):
        if self.agent_id not in env.possible_agents:
            raise ValueError("hybrid_pp_ftg requires a valid agent_id")
        self.env = env
        params = env.params
        self.wheelbase = float(params["lf"] + params["lr"])
        self.half_width = float(params["width"]) / 2
        self.front = float(params["length"]) / 2 + self.margin
        self.dt = float(env.timestep) * int(getattr(env, "_control_repeat", 1))
        actuator = params.get("wheel_actuators", {})
        self.steer_min = float(actuator.get("steering_min", params.get("s_min", -0.4189)))
        self.steer_max = float(actuator.get("steering_max", params.get("s_max", 0.4189)))
        cap = (actuator["wheel_speed_max"] * actuator["wheel_radius"]
               if actuator else params.get("v_max", self.max_speed))
        self.speed_limit = min(self.max_speed, float(cap))
        self.actuator_delay = float(actuator.get("steering_time_constant", 0.))
        self._map_source = None
        self._scan_key = None
        self.reset()
        self._map()

    def reset(self):
        self.mode = "pp"
        self._clear_count = 0
        self._nearest = None
        self._steering = None
        self._speed = None
        self._gap_angle = None
        self._mu = None
        self.last_plan = {}

    def _map(self):
        source = self.env.centerline_points
        if source is not None and source is self._map_source:
            return
        if source is None:
            raise ValueError("hybrid_pp_ftg requires a centerline")
        self.geometry = prepare_centerline_geometry(source)
        if not self.geometry.valid:
            raise ValueError("hybrid_pp_ftg requires a nonzero centerline")
        self._map_source = source  # Keep source alive across map switches.
        self.reset()

    def _points(self, arc):
        g = self.geometry
        arc = np.asarray(arc)
        arc = arc % g.total_length if g.closed else np.clip(arc, 0, g.total_length)
        indices = np.clip(np.searchsorted(g.arc_lengths, arc, side="right") - 1,
                          0, len(g.segment_lengths) - 1)
        fractions = (arc - g.arc_lengths[indices]) / g.segment_lengths[indices]
        return g.segment_starts[indices] + fractions[..., None] * g.segment_vectors[indices]

    def _scan(self, obs):
        scan = np.asarray(obs.get("lidar", obs.get("scans", [])), dtype=np.float64)
        if scan.ndim != 1 or scan.size < 3:
            return None
        # Use simulator angles, including its exact discretization and sensor offset.
        sensor = self.env.sim.agents[self.env.possible_agents.index(self.agent_id)]
        key = (scan.size, id(sensor))
        if key != self._scan_key:
            self._angles = np.array(sensor.scan_angles, dtype=np.float64)
            if self._angles.shape != scan.shape:
                raise ValueError("hybrid_pp_ftg scan does not match sensor geometry")
            self._directions = np.column_stack((np.cos(self._angles), np.sin(self._angles)))
            self._angle_step = float(np.max(np.diff(self._angles)))
            self._scan_key = key
        # Invalid returns represent unknown space. Positive infinity means no return.
        limit = float(self.env.lidar_range)
        ranges = np.clip(np.nan_to_num(scan, nan=0., posinf=limit, neginf=0.), 0., limit)
        hits = ranges < limit - 0.01
        points = ranges[hits, None] * self._directions[hits]
        points[:, 0] += float(self.env.lidar_dist)
        # Half a beam's angular width covers sparse 108-beam scans conservatively.
        radii = self.half_width + self.margin + ranges[hits] * self._angle_step * 0.5
        return points, radii

    def _traffic_envelope(self, pose, rotation, points, radii):
        """A short constant-velocity envelope catches cars merging side by side.

        LiDAR remains the obstacle source for walls/unknown objects. Public
        vehicle states cover nearby cars, including retired stationary wrecks.
        Four positions per neighbor keep this conservative check inexpensive.
        """
        extra = []
        self._traffic = []
        times = np.linspace(0., min(self.reaction_time, 0.75), 4)
        for aid in self.env.possible_agents:
            if aid == self.agent_id:
                continue
            state = self.env.get_agent_state(aid)
            relative = state.pose[:2] - pose[:2]
            if np.linalg.norm(relative) > self.env.lidar_range or (relative @ rotation)[0] < 0.:
                continue
            heading = state.pose[2]
            vx, vy = state.velocity
            world_velocity = np.array([np.cos(heading) * vx - np.sin(heading) * vy,
                                       np.sin(heading) * vx + np.cos(heading) * vy])
            self._traffic.append((relative @ rotation, world_velocity @ rotation, heading - pose[2]))
            extra.append((relative + times[:, None] * world_velocity) @ rotation)
        if extra:
            extra = np.concatenate(extra)
            radius = 2 * self.half_width + self.margin
            points = np.vstack((points, extra))
            radii = np.r_[radii, np.full(len(extra), radius)]
        return points, radii

    def _traffic_emergency(self, velocity, yaw):
        # A time-aligned short sweep catches closing traffic while steering is
        # still lagging. Only cars ahead participate, so the leader can leave.
        times = np.linspace(0., self.reaction_time + 0.2, 8)
        theta = yaw * times
        vx, vy = velocity
        if abs(yaw) > 1e-5:
            ego = np.column_stack(((vx * np.sin(theta) + vy * (np.cos(theta) - 1)) / yaw,
                                   (vx * (1 - np.cos(theta)) + vy * np.sin(theta)) / yaw))
        else:
            ego = times[:, None] * velocity
        half_length = float(self.env.params["length"]) / 2
        hl, hw = half_length + self.margin / 2, self.half_width + self.margin / 2
        co, si = np.cos(theta), np.sin(theta)
        for relative, other_velocity, other_heading in self._traffic:
            predicted = relative + times[:, None] * other_velocity - ego
            distances = np.linalg.norm(predicted, axis=1)
            delta = other_heading - theta
            cd, sd = np.abs(np.cos(delta)), np.abs(np.sin(delta))
            # Separating-axis test for two padded rectangles at each time.
            ego_x = np.abs(predicted[:, 0] * co + predicted[:, 1] * si)
            ego_y = np.abs(-predicted[:, 0] * si + predicted[:, 1] * co)
            other_co, other_si = np.cos(other_heading), np.sin(other_heading)
            other_x = np.abs(predicted[:, 0] * other_co + predicted[:, 1] * other_si)
            other_y = np.abs(-predicted[:, 0] * other_si + predicted[:, 1] * other_co)
            overlap = ((ego_x < hl + hl * cd + hw * sd) & (ego_y < hw + hl * sd + hw * cd)
                       & (other_x < hl + hl * cd + hw * sd) & (other_y < hw + hl * sd + hw * cd))
            if np.any(overlap) and distances[-1] < distances[0]:
                return True
        return False

    def _clearance(self, path, distances, points, radii):
        if not len(points):
            return float("inf")
        hit = np.any(np.sum((path[:, None, :] - points[None, :, :]) ** 2, axis=2)
                     < radii[None, :] ** 2, axis=1)
        indices = np.flatnonzero(hit)
        return float(distances[indices[0]]) if len(indices) else float("inf")

    def _arc(self, curvature, distances):
        if abs(curvature) < 1e-6:
            return np.column_stack((distances, np.zeros_like(distances)))
        theta = distances * curvature
        return np.column_stack((np.sin(theta) / curvature,
                                (1 - np.cos(theta)) / curvature))

    def _motion_path(self, steer, speed, distances):
        """Bounded kinematic sweep through actuator lag for collision checks."""
        desired = np.tan(steer) / self.wheelbase
        lag_distance = max(speed, 0.5) * self.actuator_delay
        if lag_distance < 1e-6:
            return self._arc(desired, distances)
        heading = (desired * distances + (self._initial_curvature - desired) * lag_distance
                   * (-np.expm1(-distances / lag_distance)))
        directions = np.column_stack((np.cos(heading), np.sin(heading)))
        increments = np.diff(distances)[:, None] * (directions[1:] + directions[:-1]) / 2
        return np.vstack((np.zeros(2), np.cumsum(increments, axis=0)))

    def _gap(self, bearing, speed, points, radii, obstacle_distance, timing=None):
        horizon = float(self.env.lidar_range)
        # Inflated radial clearance is equivalent to extending every obstacle's
        # angular disparity by the vehicle half-width plus a safety margin.
        candidates = np.flatnonzero(np.abs(self._angles) <= self.gap_fov / 2)
        directions = self._directions[candidates]
        if len(points):
            along = directions @ points.T
            across = directions[:, :1] * points[None, :, 1] - directions[:, 1:] * points[None, :, 0]
            blocking = (np.abs(across) < radii) & (along >= 0)
            radial = np.min(np.where(blocking, np.maximum(0, along - radii), horizon), axis=1)
        else:
            radial = np.full(len(candidates), horizon)
        # The gap must extend beyond the obstacle that triggered avoidance.
        # A fixed short threshold would label the space in front of a distant
        # parked car as a gap and defer the pass until it is too close to turn.
        choices = []
        thresholds = [self.min_gap_distance + self.front]
        if np.isfinite(obstacle_distance):
            thresholds.append(max(thresholds[0], min(horizon - 0.1, obstacle_distance + self.front + 0.5)))
        for tier, threshold in enumerate(thresholds):
            free = radial >= threshold
            # Identify contiguous navigable gaps; don't chase an isolated beam.
            edges = np.diff(np.r_[False, free, False].astype(np.int8))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            for start, end in zip(starts, ends):
                if end - start < 2:
                    continue
                ids = np.arange(start, end)
                angle = self._angles[candidates[ids]]
                cost = np.abs(angle - bearing) + 0.15 / radial[ids]
                if self._gap_angle is not None:
                    cost += 0.25 * np.abs(angle - self._gap_angle)
                # Check interior alternatives as well as the preferred ray.
                choices.extend((tier, int(local)) for local in
                               (ids[np.argmin(cost)], ids[len(ids) // 2],
                                ids[len(ids) // 4], ids[3 * len(ids) // 4]))
        if not choices:
            return 0., 0., None
        best_by_tier = {}
        distances = np.linspace(0., horizon, 40)
        if timing is not None:
            timing.context["gap_candidates"] = len(dict.fromkeys(choices))
        for tier, local in dict.fromkeys(choices):
            angle = float(self._angles[candidates[local]])
            target_distance = max(1., min(radial[local], self.lookahead_base + self.lookahead_time * speed))
            steer = float(np.clip(np.arctan(2 * self.wheelbase * np.sin(angle) / target_distance),
                                  self.steer_min, self.steer_max))
            clearance = self._clearance(self._motion_path(steer, speed, distances),
                                        distances, points, radii) - self.front
            score = min(clearance, horizon) - 1.5 * abs(angle - bearing)
            if self._gap_angle is not None:
                score -= 0.5 * abs(angle - self._gap_angle)
            if tier not in best_by_tier or score > best_by_tier[tier][0]:
                best_by_tier[tier] = (score, steer, max(0., clearance), angle)
        # Prefer a gap that passes the blocking object. On a bend, radial rays
        # can terminate at the outer wall; fall back to nearer gaps when needed.
        best = best_by_tier.get(1)
        if best is None or best[2] < self.min_gap_distance:
            best = best_by_tier.get(0, best)
        return best[1:]

    def act(self, obs, deterministic=False):
        profile = self.profile_hybrid
        if profile is None:
            return self._act(obs)
        timing = _DecisionTiming()
        action = self._act(obs, timing)
        timing.mark("other")
        # Internal timing excludes adapter conversion and reporting/aggregation.
        seconds = timing.boundary - timing.started
        context = timing.context
        profile(dict(agent_id=self.agent_id, seconds=seconds, stages=timing.stages,
                     stage_calls=timing.stage_calls, mode=("invalid" if context["invalid_state"] else self.mode),
                     target_speed=float(action[1]), abs_steering=abs(float(action[0])),
                     stopped=int(action[1] <= 1e-6),
                     braking=int(action[1] < context.get("input_speed", 0.) - 1e-3),
                     **context))
        return action

    def _act(self, obs, timing=None):
        if self.env is None:
            raise ValueError("Call set_env before using hybrid_pp_ftg")
        if timing is not None:
            timing.context["map_changed"] = int(self.env.centerline_points is not self._map_source)
        self._map()
        if timing is not None:
            timing.context["cold_start"] = int(self._steering is None)
            previous_mode = self.mode
            timing.mark("map_preparation")
        pose = np.asarray(obs["pose"], dtype=np.float64)
        velocity = np.asarray(obs["velocity"], dtype=np.float64)
        if pose.shape != (3,) or velocity.shape != (2,) or not np.isfinite(np.r_[pose, velocity]).all():
            self.reset()
            if timing is not None:
                timing.context["invalid_state"] = 1
            return np.zeros(2, dtype=np.float32)
        speed = max(0., float(velocity[0]))
        if timing is not None:
            timing.context["input_speed"] = speed
        projection = project_to_centerline(self.geometry, pose[:2], pose[2],
                                           last_index=self._nearest)
        # Recover after teleports or larger avoidance excursions.
        if abs(projection.lateral_error) > 2.:
            projection = project_to_centerline(self.geometry, pose[:2], pose[2])
        self._nearest = projection.segment_index
        lookahead = min(self.lookahead_max, self.lookahead_base + self.lookahead_time * speed)
        rotation = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                             [np.sin(pose[2]), np.cos(pose[2])]])
        # Predict through the steering actuator lag with measured yaw. This is
        # a closed-form constant-turn extrapolation, not an online optimizer.
        yaw = float(obs.get("angular_velocity", 0.))
        measured_steering = float(obs.get("steering_angle", 0.))
        if not np.isfinite([yaw, measured_steering]).all():
            self.reset()
            if timing is not None:
                timing.context["invalid_state"] = 1
            return np.zeros(2, dtype=np.float32)
        self._initial_curvature = (yaw / speed if speed > 0.5
                                   else np.tan(measured_steering) / self.wheelbase)
        delay = self.actuator_delay
        heading_change = yaw * delay
        travel = speed * delay
        predicted_local = (np.array([np.sin(heading_change), 1 - np.cos(heading_change)]) * speed / yaw
                           if abs(yaw) > 1e-5 else np.array([travel, 0.]))
        predicted = pose[:2] + predicted_local @ rotation.T
        predicted_heading = pose[2] + heading_change
        predicted_rotation = np.array([[np.cos(predicted_heading), -np.sin(predicted_heading)],
                                       [np.sin(predicted_heading), np.cos(predicted_heading)]])
        world_target = self._points(projection.arc_length + travel + lookahead)
        target = (world_target - predicted) @ predicted_rotation
        actual_target = (world_target - pose[:2]) @ rotation
        bearing = float(np.arctan2(actual_target[1], actual_target[0]))
        curvature = 2 * target[1] / max(float(target @ target), 0.01)
        steer = float(np.clip(np.arctan(self.wheelbase * curvature), self.steer_min, self.steer_max))
        if timing is not None:
            timing.mark("pure_pursuit")

        horizon = min(float(self.env.lidar_range), max(lookahead + 1.,
                      speed * self.reaction_time + speed ** 2 / (2 * self.braking) + self.front + 1.))
        distances = np.linspace(0., horizon, max(3, int(np.ceil(horizon / self.path_spacing)) + 1))
        if timing is not None:
            timing.context["path_samples"] = len(distances)
        arcs = projection.arc_length + distances
        route = self._points(arcs)
        spacing = self.curvature_spacing
        before, after = self._points(arcs - spacing), self._points(arcs + spacing)
        u, v = route - before, after - route
        curve = np.abs(2 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) /
                       np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
                                  * np.linalg.norm(after - before, axis=1), 1e-6))
        # The public global-state metadata carries the sampled episode grip.
        if self._mu is None:
            physics = self.env.get_global_state().metadata.get("physics", {})
            self._mu = float(physics.get("mu", self.env.params.get("mu", 1.)))
        mu = self._mu
        if not np.isfinite(mu) or mu < 0:
            if timing is not None:
                timing.context["invalid_state"] = 1
                timing.mark("speed_planning")
            return np.zeros(2, dtype=np.float32)
        lateral_limit = min(self.lateral_acceleration, max(0., mu) * 9.81 * 0.6)
        corner_speeds = np.sqrt(lateral_limit / np.maximum(curve, 1e-4))
        braking_distances = np.maximum(0., distances - speed * self.reaction_time)
        target_speed = min(self.speed_limit, float(np.min(np.sqrt(corner_speeds ** 2 + 2 * self.braking * braking_distances))),
                           np.sqrt(lateral_limit / max(abs(curvature), 1e-4)))
        # Body-frame lateral slip and yaw error reduce speed before loss of control.
        stability = max(abs(velocity[1]) / max(speed, 1.), abs(yaw - speed * curvature) * 0.15)
        target_speed /= 1 + 2 * max(0., stability - 0.15)
        if abs(projection.lateral_error) > 0.5:
            target_speed = min(target_speed, 4. / (1 + abs(projection.lateral_error)))
        if not self.geometry.closed:
            target_speed = min(target_speed, np.sqrt(2 * self.braking * max(0., self.geometry.total_length - projection.arc_length - self.front)))
        if timing is not None:
            timing.mark("speed_planning")

        scan = self._scan(obs)
        if timing is not None:
            timing.context["lidar_hits"] = len(scan[0]) if scan is not None else 0
            timing.mark("lidar_preprocessing")
        emergency = False
        clearance, gap_angle = 0., None
        if scan is None:
            self.mode, self._clear_count = "ftg", 0
            target_speed = 0.
            if timing is not None:
                timing.context["missing_scan"] = 1
        else:
            points, radii = self._traffic_envelope(pose, rotation, *scan)
            emergency = self._traffic_emergency(velocity, yaw)
            if timing is not None:
                timing.context["traffic_count"] = len(self._traffic)
                timing.mark("traffic_prediction")
            local_route = (route - pose[:2]) @ rotation
            clearance = self._clearance(local_route, distances, points, radii) - self.front
            near_distances = np.linspace(0., min(horizon, max(0.75, speed * (self.actuator_delay + 0.2) + self.front)), 20)
            immediate = self._clearance(self._motion_path(steer, speed, near_distances), near_distances, *scan) - self.front
            clearance = min(clearance, immediate)
            enter_distance = speed * self.reaction_time + speed ** 2 / (2 * self.braking) + 0.6
            blocked = clearance < enter_distance
            if blocked:
                self.mode, self._clear_count = "ftg", 0
            elif self.mode == "ftg":
                self._clear_count = self._clear_count + 1 if clearance > enter_distance + self.exit_margin else 0
                if self._clear_count >= self.clear_steps:
                    self.mode, self._gap_angle = "pp", None
            if timing is not None:
                timing.context["blocked"] = int(blocked)
                timing.mark("path_checks")
            if self.mode == "ftg":
                # Forecasts trigger avoidance early. Gap geometry uses current
                # LiDAR occupancy, so cars do not treat a neighbor's entire
                # future envelope as a stationary barrier and deadlock.
                steer, clearance, gap_angle = self._gap(bearing, speed, *scan, clearance, timing=timing)
                self._gap_angle = gap_angle
                target_speed = min(target_speed, self.gap_speed,
                                   np.sqrt(lateral_limit / max(abs(np.tan(steer) / self.wheelbase), 1e-4)))
                # Smooth acceleration; braking safety limits apply immediately.
                safe_speed = max(0., -self.braking * self.reaction_time +
                                 np.sqrt((self.braking * self.reaction_time) ** 2 + 2 * self.braking * max(0., clearance)))
                target_speed = min(target_speed, safe_speed) if gap_angle is not None else 0.
                if timing is not None:
                    timing.context["no_gap"] = int(gap_angle is None)
                    timing.mark("gap_selection")

        if timing is not None:
            # Branch bookkeeping not included in the actuator smoothing stage.
            timing.mark("other")
        previous = measured_steering if self._steering is None else self._steering
        alpha = -np.expm1(-self.dt / self.steering_time_constant)
        steer = previous + np.clip(alpha * (steer - previous), -self.steering_rate * self.dt, self.steering_rate * self.dt)
        previous_speed = speed if self._speed is None else self._speed
        target_speed = min(target_speed, previous_speed + self.acceleration * self.dt)
        if self.mode == "pp":
            target_speed = max(target_speed, previous_speed - self.braking * self.dt)
        if emergency:
            target_speed = 0.
        action = np.array([np.clip(steer, self.steer_min, self.steer_max),
                           np.clip(target_speed, 0., self.speed_limit)], dtype=np.float32)
        self._steering, self._speed = map(float, action)
        self.last_plan = dict(mode=self.mode, clearance=float(clearance), gap_angle=gap_angle,
                              target_speed=float(action[1]), steering=float(action[0]),
                              lateral_error=projection.lateral_error, emergency_brake=emergency)
        if timing is not None:
            timing.context.update(emergency_brake=int(emergency),
                                  mode_switch=int(self.mode != previous_mode))
            timing.mark("smoothing")
        return action
