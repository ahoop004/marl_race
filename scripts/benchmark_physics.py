"""Short deterministic P1 serial baselines; no learner, training, or parallel work.

Run from the repository: python3 scripts/benchmark_physics.py --output outputs/physics_p1
Use --source-root with an archived revision for an uninstrumented comparison.
The harness/configuration come from this checkout; simulation imports come only
from source-root. Reset, compilation, hashing and serialization are not timed as
steady-state steps. Repeat both modes in alternating order to reduce drift.
"""
import argparse
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
from time import perf_counter, process_time

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "7d65c76fadff9081ce59c4a42c7400b892adc92e"


def digest(value, result=None):
    """Exact recursive evidence, including arrays, events, rewards and task masks."""
    result = result or hashlib.sha256()
    result.update(type(value).__name__.encode() + b"\0")
    if isinstance(value, np.ndarray):
        result.update(str((value.dtype.str, value.shape)).encode())
        result.update(value.tobytes())
    elif is_dataclass(value):
        for field in fields(value):
            result.update(field.name.encode())
            digest(getattr(value, field.name), result)
    elif isinstance(value, Mapping):
        result.update(str(len(value)).encode() + b"\0")
        for key in sorted(value):
            digest(key, result)
            digest(value[key], result)
    elif isinstance(value, (tuple, list)):
        result.update(str(len(value)).encode() + b"\0")
        for row in value:
            digest(row, result)
    else:
        result.update(repr(value).encode())
    return result


def synthetic_map(destination):
    # 40 m square at existing common 5 cm occupancy resolution, closed border.
    pixels = np.full((800, 800), 255, dtype=np.uint8)
    pixels[[0, -1], :] = 0
    pixels[:, [0, -1]] = 0
    Image.fromarray(pixels).save(destination / "arena.png")
    path = destination / "arena.yaml"
    path.write_text(yaml.safe_dump(dict(image="arena.png", resolution=.05,
                                      origin=[0., 0., 0.], negate=0,
                                      occupied_thresh=.65, free_thresh=.196)))
    return path


def build_case(name, map_path, source_root, steps):
    from physics.simulaton import Simulator
    if name == "B6":
        from core.scenario import load_and_expand_scenario
        from core.task_builder import create_race_task
        # Keep the production MAPPO observation/reward/fixed hybrid controllers.
        # Fixed nominal grip removes episode randomization for matched baselines.
        config = load_and_expand_scenario(str(ROOT / "scenarios/mappo_2v2_completion_scratch.yaml"))
        config["environment"].update(timestep=.05, action_repeat=1, max_steps=steps,
                                     map_bundles_train=["circle_map"], map_cycle="per_episode",
                                     map_pick="round_robin", render=False)
        config["environment"]["friction"]["train"] = dict(mode="fixed", mu=1.0489)
        task = create_race_task(config, scenario_dir=source_root / "scenarios")
        def reset():
            task.reset(seed=42, options={"map_episode_index": 2, "spawn_episode_index": 3})
        def advance(index):
            actions = {aid: np.array([.02 * np.sin(index * .2), .1], dtype=np.float32)
                       for aid in task.agents}
            return task.step(actions)
        return task.env.sim, task, reset, advance, config
    count = {"B1": 1, "B2": 2, "B3": 4, "B4": 2, "B5": 4}[name]
    params = yaml.safe_load((ROOT / "configs/vehicle/combined_slip.yaml").read_text())
    sim = Simulator(params, count, 42, time_step=.05, num_beams=108)
    sim.set_map(str(map_path), ".png")
    poses = np.array([[5., 5. + 2 * i, 0.] for i in range(count)])
    velocities = np.full(count, 2.)
    if name == "B4":
        poses = np.array([[5., 5., 0.], [6.4, 5., np.pi]])
    elif name == "B5":
        poses = np.array([[5., 5., 0.], [6.1, 5., 0.], [7.5, 5., 0.], [8.3, 5., np.pi]])
        velocities = np.array([4., 2., 0., 2.])
    def reset():
        sim.reseed(42)
        sim.reset(poses, velocities=velocities)
    def advance(index):
        actions = np.column_stack((np.zeros(count), velocities / .05))
        if name in {"B1", "B2", "B3"}:
            actions[:, 0] = .04 * np.sin(index * .2)
        return sim.step(actions)
    config = dict(vehicle_params=params, timestep=.05, lidar_beams=108,
                  poses=poses.tolist(), velocities=velocities.tolist(), seed=42,
                  collision_steps=12, free_steps=steps,
                  map=dict(size_m=40, resolution=.05, closed_border=True),
                  steering="0.04*sin(0.2*step) for B1-B3; zero for contacts",
                  wheel_reference="initial speed / wheel_radius")
    return sim, None, reset, advance, config


def run_rollout(sim, task, reset, advance, steps, profile=None, *, capture=False):
    reset()  # reset is always outside steady-state measurement
    evidence, trajectory, contacts = hashlib.sha256(), [], []
    wall = cpu = 0.
    decisions = physics_steps = 0
    context = profile.attach(sim, task=task) if profile else nullcontext()
    if profile:
        profile.clear()
    with context:
        for index in range(steps):
            started, cpu_started = perf_counter(), process_time()
            output = advance(index)
            cpu += process_time() - cpu_started
            wall += perf_counter() - started
            physics_steps += output.physics_steps if task else 1
            decisions += len(output.decisions) if task else 0
            # Capture/parity work never enters timing envelopes.
            if capture:
                state = np.stack([car.physics_state.copy() for car in sim.agents])
                digest((output, state, sim.collision_idx,
                        getattr(sim, "collision_times", None)), evidence)
                trajectory.append(state.tolist())
                times = getattr(sim, "collision_times", np.full(sim.num_agents, np.inf))
                contacts.append(dict(flags=sim.collisions.tolist(), indices=sim.collision_idx.tolist(),
                                     times=[float(t) if np.isfinite(t) else None for t in times]))
            if task and task.episode_done:
                break
    record = dict(wall_seconds=wall, cpu_seconds=cpu, cpu_percent=100 * cpu / wall,
                  environment_steps=index + 1, vehicle_steps=(index + 1) * sim.num_agents,
                  physics_steps=physics_steps, policy_decisions=decisions,
                  environment_steps_per_second=(index + 1) / wall,
                  physics_steps_per_second=physics_steps / wall,
                  vehicle_steps_per_second=(index + 1) * sim.num_agents / wall,
                  seconds_per_vehicle_step=wall / ((index + 1) * sim.num_agents),
                  policy_decisions_per_second=decisions / wall if task else None,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if capture:
        record.update(digest=evidence.hexdigest(), trajectory=trajectory, contacts=contacts)
    if profile:
        record["profile"] = profile.export(wall)
    return record


def kernel_baselines(sim, *, iterations=20000, repeats=7):
    """Standalone kernel costs, never added to rollout stages or percentages.

    Vary inputs in compiled loops to amortize dispatch without clock reads in
    kernels. RHS includes load equilibrium plus tire forces; its nested costs
    cannot be subtracted from these different workloads to infer stage shares.
    """
    from numba import njit
    from physics.dynamic_models import combined_slip_dynamics, sample_wheel_actuators, write_rk4_trace
    from physics.tire_models import mf61_tire_force
    from physics.collision_models import _trace_bounds, _separation
    car = sim.agents[0]
    vehicle = car._physics
    actuators = vehicle._actuators
    params = vehicle._dynamics_params
    states = np.array([[5., 5., .1, 2., .03, .1],
                       [5.1, 5.1, .2, 2.2, -.02, -.1],
                       [5.2, 5.2, .3, 2.1, .01, .05]])
    act = np.array([.04, 44.])
    reference = np.array([.1, 50.])
    trace = car._pose_trace.copy()
    other = trace.copy()
    other[:, 0, 1] += .4
    extents = np.array([car.geometry.front, car.geometry.rear, car.geometry.left, car.geometry.right])

    @njit
    def batch(kind, count):
        checksum = 0.
        out = np.zeros((4, 6))
        for index in range(count):
            state = states[index % len(states)]
            if kind == 0:
                value = mf61_tire_force(state[3], state[4], 2.2, 19., params[5], params[8], params[6])
                checksum += value[0]
            elif kind == 1:
                rhs, _ = combined_slip_dynamics(state, act, params)
                checksum += rhs[3]
            elif kind == 2:
                value = sample_wheel_actuators(act, reference, (index % 51) * .001,
                                              actuators_constants, actuators_rates)
                checksum += value[1]
            elif kind == 3:
                write_rk4_trace(out, state, states[0], states[1], states[2], state, .001)
                checksum += out[2, 3]
            elif kind == 4:
                lower, upper = _trace_bounds(trace, extents)
                checksum += lower[0] + upper[0]
            else:
                checksum += _separation(trace, extents, other, extents,
                                        (index % 51) / 50., np.inf, np.inf, 0., 0.)
        return checksum

    actuators_constants, actuators_rates = actuators._time_constants, actuators._rate_limits
    result = {}
    for kind, name in enumerate(("mf61_single_axle", "rhs_with_load_solve", "actuator_pair_sample",
                                 "dense_trace_segment", "swept_aabb", "sat_separation")):
        started = perf_counter()
        batch(kind, 1)
        warmup = perf_counter() - started
        values, checksum = [], 0.
        for _ in range(repeats):
            started = perf_counter()
            checksum += batch(kind, iterations)
            values.append((perf_counter() - started) / iterations)
        result[name] = dict(seconds_per_call=values, median_seconds_per_call=float(np.median(values)),
                            checksum=checksum, warmup_seconds=warmup, calls_per_repeat=iterations)
    return result


def main():
    suite_started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--source-revision", default=None, help="Required for an archived source without .git")
    parser.add_argument("--scenarios", nargs="+", choices=[f"B{i}" for i in range(1, 7)],
                        default=[f"B{i}" for i in range(1, 7)])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--kernels", action="store_true", help="Also measure standalone corrected kernels")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    args.output = args.output.resolve()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    os.chdir(source_root)
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    args.output.mkdir(parents=True, exist_ok=True)
    def git(*command):
        return subprocess.check_output(["git", "-C", str(source_root), *command], text=True).strip()
    revision = args.source_revision or git("rev-parse", "HEAD")
    metadata = dict(source_revision=revision, branch=git("branch", "--show-current") if (source_root / ".git").exists() else None,
                    source_root=str(source_root), python=sys.version, platform=platform.platform(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)), steps=args.steps, repeats=args.repeats,
                    command=sys.argv, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    methodology="alternating disabled/enabled; compilation/warmup/reset/capture excluded from throughput; exclusive stages")
    # Save the exact instrumentation patch and hashes: HEAD alone is insufficient.
    if (source_root / ".git").exists():
        metadata["git_status"] = git("status", "--short")
        patch = git("diff", "--binary", "HEAD")
        (args.output / "instrumentation.patch").write_text(patch + "\n")
        for relative in git("ls-files", "--modified", "--others", "--exclude-standard").splitlines():
            path = source_root / relative
            if path.is_file() and path.suffix == ".py":
                destination = args.output / "source_snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    metadata["source_file_sha256"] = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source_root / "src").rglob("*.py"))}
    import numba
    import scipy
    metadata["versions"] = dict(numpy=np.__version__, numba=numba.__version__, scipy=scipy.__version__)
    metadata["cpu_model"] = next((line.split(":", 1)[1].strip() for line in
                                  Path("/proc/cpuinfo").read_text().splitlines()
                                  if line.startswith("model name")), platform.processor())
    metadata["rss_scope"] = "process lifetime peak, includes imports/JIT; not incremental environment memory"
    report = dict(metadata=metadata, scenarios={})
    map_path = synthetic_map(args.output)
    for name in args.scenarios:
        started = perf_counter()
        sim, task, reset, advance, config = build_case(name, map_path, source_root, args.steps)
        scanner = sim.scan_simulator
        sensor_settings = {key: getattr(scanner, key) for key in
                           ("num_beams", "fov", "eps", "theta_dis", "max_range", "map_resolution")}
        sensor_settings["default_noise_stddev"] = .01
        import physics.simulaton
        if not Path(physics.simulaton.__file__).resolve().is_relative_to(source_root):
            raise RuntimeError("Simulator imported from the wrong source tree")
        steps = min(args.steps, 12) if name in {"B4", "B5"} else args.steps
        profile = None
        if not args.no_profile:
            from physics.profiling import SerialPhysicsProfile
            profile = SerialPhysicsProfile()
        try:
            # Warm both Numba specializations through the complete workload.
            plain_evidence = run_rollout(sim, task, reset, advance, steps, capture=True)
            enabled_evidence = run_rollout(sim, task, reset, advance, steps, profile, capture=True) if profile else None
            if enabled_evidence and plain_evidence["digest"] != enabled_evidence["digest"]:
                raise AssertionError(f"{name}: instrumentation changed rollout")
            startup = perf_counter() - started
            rows = {"disabled": [], "enabled": []}
            for repeat in range(args.repeats):
                order = [False, True] if repeat % 2 == 0 else [True, False]
                for enabled in order:
                    if enabled and profile is None:
                        continue
                    rows["enabled" if enabled else "disabled"].append(
                        run_rollout(sim, task, reset, advance, steps, profile if enabled else None))
            report["scenarios"][name] = dict(config=config, sensor_settings=sensor_settings,
                                             startup_warmup_seconds=startup,
                                             parity_digest=plain_evidence["digest"],
                                             profile_bit_identical=bool(enabled_evidence),
                                             evidence=plain_evidence, runs=rows)
            if args.kernels and name == "B1":
                report["standalone_kernels_do_not_sum_with_stages"] = kernel_baselines(sim)
            print(f"{name}: " + ", ".join(f"{mode}={np.median([r['environment_steps_per_second'] for r in values]):.1f} steps/s"
                                          for mode, values in rows.items() if values), flush=True)
        finally:
            if task:
                task.close()
        report["metadata"]["suite_wall_seconds"] = perf_counter() - suite_started
        (args.output / "results.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
