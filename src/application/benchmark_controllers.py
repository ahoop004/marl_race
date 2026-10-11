"""Fixed-opponent quality and throughput benchmark using the real RaceTask.

Run with PYTHONPATH=src python3 -m application.benchmark_controllers --help.
No policy inference, training, rendering, or external logging is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from core.configuration import PROJECT_ROOT
from core.scenario import load_and_expand_scenario
from core.setup import create_environment_setup
from tasks.race_task import RaceTask
from utils.centerline import prepare_centerline_geometry


class TimedController:
    def __init__(self, controller):
        self.controller = controller
        self.times = []
        self.errors = []

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def act(self, obs):
        start = perf_counter()
        try:
            return self.controller.act(obs)
        except Exception as exc:
            self.errors.append(repr(exc))
            raise
        finally:
            self.times.append(perf_counter() - start)


class StationaryController:
    def act(self, obs):
        return np.zeros(2, dtype=np.float32)


def _obstacle_poses(env):
    """Centerline-aligned ego and parked car 8 m ahead, same for both controllers."""
    g = prepare_centerline_geometry(env.centerline_points)
    poses = []
    for arc in (0., 8.):
        index = int(np.searchsorted(g.arc_lengths, arc, side="right") - 1)
        vector = g.segment_vectors[index]
        xy = g.segment_starts[index] + vector * ((arc - g.arc_lengths[index]) / g.segment_lengths[index])
        poses.append([*xy, np.arctan2(vector[1], vector[0])])
    return {"poses": np.asarray(poses, dtype=np.float32)}


def run_case(controller, map_id, seed, case, *, max_steps=8000, laps=1,
             max_speed=8., warmup_steps=20, overrides=()):
    config = load_and_expand_scenario("scenarios/render/hybrid_pp_ftg.yaml", overrides=[
        f"controllers={controller}", f"controllers.{controller}.max_speed={max_speed}",
        "environment.render=false", "environment.centerline_render=false",
        f"environment.max_speed={max_speed}", f"environment.max_steps={max_steps}",
        f"environment.target_laps={laps}", f"environment.map_bundles=[{map_id}]",
        f"environment.map_bundles_train=[{map_id}]", f"environment.map_bundles_eval=[{map_id}]",
        *overrides,
    ])
    count = {"solo": 1, "traffic": 4, "stationary": 2}[case]
    template = config["agents"]["car_0"]
    config["agents"] = {f"car_{i}": deepcopy(template) for i in range(count)}
    env, controllers = create_environment_setup(config, scenario_dir=PROJECT_ROOT / "scenarios", mode="eval")
    if case == "stationary":
        controllers["car_1"] = StationaryController()
    timed = {aid: TimedController(value) for aid, value in controllers.items()
             if case != "stationary" or aid != "car_1"}
    controllers.update(timed)
    task = RaceTask(env, policy_agents=[], fixed_controllers=controllers,
                    obs_composers={}, reward_composers={}, action_composers={})
    options = _obstacle_poses(env) if case == "stationary" else None
    try:
        # Load physics/LiDAR JIT caches before the seeded measured episode.
        task.reset(seed=seed, options=options)
        task.step({})
        snapshot = task.reset(seed=seed, options=options)
        for value in timed.values():
            value.times.clear()
            value.errors.clear()
        samples = {aid: {"speed": [], "lateral_speed": [], "yaw_rate": [],
                         "lateral_error": [], "steering_change": [], "ftg_steps": 0,
                         "mode_switches": 0, "emergency_brake_steps": 0,
                         "track_limit_steps": 0, "track_limits_available": False}
                   for aid in timed}
        previous_mode, previous_steering = {}, {}
        elapsed = steady_elapsed = 0.
        active_steps = 0
        for step in range(max_steps):
            active = set(snapshot.active_physical_agents) & set(timed)
            start = perf_counter()
            transition = task.step({})
            duration = perf_counter() - start
            elapsed += duration
            if step >= warmup_steps:
                steady_elapsed += duration
                active_steps += 1
            snapshot = transition.after
            for aid in active:
                state = env.get_agent_state(aid)
                row = samples[aid]
                row["speed"].append(float(state.velocity[0]))
                row["lateral_speed"].append(abs(float(state.velocity[1])))
                row["yaw_rate"].append(abs(float(state.angular_velocity)))
                info = snapshot.infos[aid]
                row["lateral_error"].append(abs(float(info.get("centerline", {}).get("d", 0.))))
                row["track_limit_steps"] += bool(info.get("track_limits", {}).get("exceeded", False))
                row["track_limits_available"] |= "track_limits" in info
                plan = getattr(timed[aid], "last_plan", {})
                mode = plan.get("mode", "mpc")
                row["ftg_steps"] += mode == "ftg"
                row["emergency_brake_steps"] += bool(plan.get("emergency_brake", False))
                row["mode_switches"] += aid in previous_mode and mode != previous_mode[aid]
                previous_mode[aid] = mode
                steering = float(transition.substeps[0].physical_actions[aid][0])
                if aid in previous_steering:
                    row["steering_change"].append(abs(steering - previous_steering[aid]))
                previous_steering[aid] = steering
            if snapshot.episode_done or not (set(snapshot.active_physical_agents) & set(timed)):
                break
        results = {}
        for aid, row in samples.items():
            info = snapshot.infos[aid]
            times = np.array(timed[aid].times[warmup_steps:]) * 1000
            results[aid] = dict(laps=int(info["lap_count"]), completed=bool(info["race_completed"]),
                terminal_reason=info["terminal_reason"], collisions=int(info["terminal_reason"] == "collision"),
                mean_speed_mps=float(np.mean(row["speed"])),
                max_lateral_speed_mps=float(max(row["lateral_speed"], default=0)),
                max_yaw_rate_rad_s=float(max(row["yaw_rate"], default=0)),
                max_lateral_error_m=float(max(row["lateral_error"], default=0)),
                mean_steering_change_rad=float(np.mean(row["steering_change"])) if row["steering_change"] else 0.,
                ftg_fraction=row["ftg_steps"] / max(1, len(row["speed"])),
                mode_switches=row["mode_switches"], emergency_brake_steps=row["emergency_brake_steps"],
                track_limit_steps=row["track_limit_steps"] if row["track_limits_available"] else None,
                controller_calls=len(timed[aid].times), controller_errors=timed[aid].errors,
                first_call_ms=timed[aid].times[0] * 1000,
                controller_mean_ms=float(np.mean(times)) if len(times) else None,
                controller_p95_ms=float(np.percentile(times, 95)) if len(times) else None)
        return dict(controller=controller, map=map_id, seed=seed, case=case,
                    max_speed_mps=max_speed, steps=step + 1, seconds=elapsed,
                    steps_per_second=(step + 1) / elapsed,
                    steady_steps_per_second=active_steps / steady_elapsed if steady_elapsed else None,
                    warmup_steps=warmup_steps, agents=results,
                    spawn=task.episode_metadata.spawn_configuration)
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", nargs="+", choices=["hybrid_pp_ftg", "racing_mpc"], default=["hybrid_pp_ftg"])
    parser.add_argument("--maps", nargs="+", default=["circle_map", "Budapest_map"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--cases", nargs="+", choices=["solo", "traffic", "stationary"], default=["solo", "traffic", "stationary"])
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-speed", type=float, default=8.)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("outputs/controller_benchmark.json"))
    args = parser.parse_args()
    if args.max_steps < 1 or args.laps < 1 or args.warmup_steps < 0 or not np.isfinite(args.max_speed) or args.max_speed <= 0:
        parser.error("steps, laps and speed must be positive; warmup must be nonnegative")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for controller in args.controllers:
        for map_id in args.maps:
            for seed in args.seeds:
                for case in args.cases:
                    result = run_case(controller, map_id, seed, case, max_steps=args.max_steps,
                        laps=args.laps, max_speed=args.max_speed, warmup_steps=args.warmup_steps,
                        overrides=args.set)
                    results.append(result)
                    print(json.dumps({key: value for key, value in result.items() if key != "spawn"}), flush=True)
                    args.output.write_text(json.dumps(dict(protocol=vars(args) | {"output": str(args.output)},
                        timing="RaceTask.step incl. adapter and snapshots; diagnostic aggregation excluded; JIT primed before reset",
                        results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
