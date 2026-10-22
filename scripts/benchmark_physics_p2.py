"""P2 serial attribution: identical physical actions through progressively fuller layers.

No training or optimization. Recording/sampling and detailed profiling are separate
from throughput runs. Live task and recorded-controller task share learner inputs.
"""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYGLET_HEADLESS", "true")
from scripts import benchmark_physics as p1
from physics.profiling import SerialPhysicsProfile

LAYERS = ("simulator", "environment", "task_replay", "task_live")


@contextmanager
def replace(owner, name, value):
    had, previous = name in vars(owner), vars(owner).get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if had:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def build_workload(map_id, steps, *, layout="spaced"):
    from agents.fixed.hybrid import HybridPPFTGAgent
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task
    config = load_and_expand_scenario(str(ROOT / "scenarios/mappo_2v2_completion_scratch.yaml"))
    config["environment"].update(max_steps=steps, timestep=.05, action_repeat=1,
                                 map_bundles_train=[map_id], map_pick="round_robin",
                                 map_cycle="per_episode", render=False)
    config["environment"]["friction"]["train"] = dict(mode="fixed", mu=1.0489)
    task = create_race_task(config, scenario_dir=ROOT / "scenarios")
    drivers = {}
    for aid in task.possible_agents:
        driver = HybridPPFTGAgent({"params": {**config["agents"]["car_2"]["params"], "agent_id": aid}})
        driver.set_env(task.env)
        drivers[aid] = driver
    geometry = next(iter(drivers.values())).geometry
    offsets = np.array([0., -9., -18., -27.]) if layout == "spaced" else np.array([0., -3., -11., -16.])
    arcs = (.05 * geometry.total_length + offsets) % geometry.total_length
    xy = next(iter(drivers.values()))._points(arcs)
    indices = np.clip(np.searchsorted(geometry.arc_lengths, arcs, side="right") - 1,
                      0, len(geometry.segment_vectors) - 1)
    tangents = geometry.segment_vectors[indices]
    poses = np.column_stack((xy, np.arctan2(tangents[:, 1], tangents[:, 0])))
    options = dict(poses=poses, map_episode_index=2, spawn_episode_index=3)
    def reset():
        snapshot = task.reset(seed=42, options=options)
        for driver in drivers.values():
            driver.reset()
        return snapshot
    return task, drivers, reset, dict(scenario=config, poses=poses.tolist(), layout=layout,
                                      track_length=geometry.total_length, seed=42)


def driver_actions(task, drivers):
    """Existing hybrid controllers generate representative deterministic learner inputs.

    Encode desired rolling speed as the existing bounded wheel-acceleration
    action contract. No controller or physics parameters are changed.
    """
    actions = {}
    for aid in task.agents:
        desired = drivers[aid].act(task._snapshot.raw_observations[aid])
        composer = task.action_composers[aid]
        integrated = next(c for c in composer._components if hasattr(c, "_speed"))
        radius = task.env.params["wheel_actuators"]["wheel_radius"]
        delta = float(desired[1]) / radius - integrated._speed
        rate = integrated._acceleration if delta >= 0 else integrated._deceleration
        bounds = task.env.action_space(aid)
        steering = 2 * (float(desired[0]) - bounds.low[0]) / (bounds.high[0] - bounds.low[0]) - 1
        actions[aid] = np.array([np.clip(steering, -1, 1), np.clip(delta / (rate * task.timestep), -1, 1)], dtype=np.float32)
    return actions


def physics_digest(sim, output):
    return p1.digest((output, np.stack([car.physics_state for car in sim.agents]),
                      sim.collision_idx, sim.collision_times,
                      [car.control_reference for car in sim.agents])).hexdigest()


def record_trace(task, drivers, reset, steps, *, sample_every=20):
    sim = task.env.sim
    reset()
    frames, samples = [], []
    observation = {}
    original = sim.step
    buffers = [np.empty((4, 8)) for _ in sim.agents]
    def capture(controls):
        observation["controls"] = controls.copy()
        output = original(controls)
        observation["physics_digest"] = physics_digest(sim, output)
        return output
    with replace(sim, "step", capture):
        for index in range(steps):
            sampled = index % sample_every == 0
            with ExitStack() as resources:
                if sampled:
                    for car, buffer in zip(sim.agents, buffers):
                        resources.enter_context(replace(car._physics, "profile_rhs_samples", buffer))
                normalized = driver_actions(task, drivers)
                output = task.step(normalized)
                if sampled:
                    for car, buffer in zip(sim.agents, buffers):
                        if not car._motion_frozen:
                            samples.extend(buffer.copy())
            frames.append(dict(normalized=normalized,
                               env_actions=output.substeps[0].physical_actions,
                               controls=observation["controls"],
                               physics_digest=observation["physics_digest"],
                               task_digest=p1.digest(output).hexdigest(),
                               state=np.stack([car.physics_state for car in sim.agents]),
                               collisions=sim.collisions.copy(),
                               active=len(task.agents)))
            if task.episode_done:
                break
    return frames, np.asarray(samples)


def trajectory_statistics(frames):
    state = np.array([f["state"] for f in frames])
    distance = np.sum(np.linalg.norm(np.diff(state[:, :, :2], axis=0), axis=2), axis=0)
    heading = np.unwrap(state[:, :, 2], axis=0)
    pair_distances = np.array([np.linalg.norm(state[:, i, :2] - state[:, j, :2], axis=1)
                               for i in range(4) for j in range(i + 1, 4)])
    return dict(steps=len(frames), simulated_seconds=len(frames) * .05,
                distance_m_per_car=distance.tolist(),
                accumulated_abs_heading_change_rad=np.sum(np.abs(np.diff(heading, axis=0)), axis=0).tolist(),
                speed_quantiles_mps=np.percentile(np.linalg.norm(state[:, :, 3:5], axis=2), [0, 50, 95, 100]).tolist(),
                yaw_rate_abs_quantiles=np.percentile(np.abs(state[:, :, 5]), [50, 95, 100]).tolist(),
                pair_distance_range_m=[float(pair_distances.min()), float(pair_distances.max())],
                collision_steps=sum(bool(np.any(f["collisions"])) for f in frames),
                final_policy_agents=frames[-1]["active"])


def replay(task, reset, frames, layer, *, profile=None, capture=False):
    sim = task.env.sim
    reset()
    index = [0]
    step_walls = []
    # Replay hooks replace only fixed controller planning. Learner action
    # composers, task observation/reward/snapshot logic remain in production use.
    with ExitStack() as resources:
        if layer == "task_replay":
            for aid, controller in task.fixed_controllers.items():
                resources.enter_context(replace(controller, "act", lambda obs, aid=aid:
                                                frames[index[0]]["env_actions"][aid].copy()))
        if profile:
            profile.clear()
            resources.enter_context(profile.attach(sim, task=task if layer.startswith("task") else None,
                                                    environment=task.env if layer == "environment" else None,
                                                    detail=True))
        observation = {}
        if capture:
            original = sim.step
            def verify(controls):
                np.testing.assert_array_equal(controls, frames[index[0]]["controls"])
                result = original(controls)
                observation["digest"] = physics_digest(sim, result)
                return result
            resources.enter_context(replace(sim, "step", verify))
        cpu_started = process_time()
        for position, frame in enumerate(frames):
            index[0] = position
            started = perf_counter()
            if layer == "simulator":
                output = sim.step(frame["controls"])
            elif layer == "environment":
                output = task.env.step(frame["env_actions"])
            else:
                output = task.step(frame["normalized"])
            step_walls.append(perf_counter() - started)
            if capture:
                if observation["digest"] != frame["physics_digest"]:
                    raise AssertionError(f"{layer}: physics diverged at {position}")
                if layer.startswith("task") and p1.digest(output).hexdigest() != frame["task_digest"]:
                    raise AssertionError(f"{layer}: task output diverged at {position}")
        cpu = process_time() - cpu_started
    wall = sum(step_walls)
    result = dict(wall_seconds=wall, step_wall_seconds=step_walls, steps=len(frames),
                  steps_per_second=len(frames) / wall,
                  validation_capture=capture,
                  cpu_seconds_including_capture=cpu,
                  peak_lifetime_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if capture:
        result["exact_physics_parity"] = True
        result["exact_task_parity"] = layer.startswith("task")
    if profile:
        result["profile"] = profile.export(wall)
    return result


def solver_breakdown(samples, params, *, repeats=7, batches=8):
    """Replay actual RHS/force inputs in separate compiled batches, without clocks in kernels.

    Nested rows are inclusive workloads, not additive percentages. Differences
    include replay/checksum overhead and are estimates, not exact self CPU time.
    """
    from numba import njit
    from physics.dynamic_models import _axle_forces, combined_slip_dynamics
    from physics.tire_models import mf61_tire_force
    samples = np.ascontiguousarray(samples)
    traces = np.zeros((len(samples), 63, 8))
    counts = np.empty(len(samples), dtype=np.int64)
    for index, sample in enumerate(samples):
        rhs, axles = combined_slip_dynamics(sample[:6], sample[6:], params, force_samples=traces[index])
        plain_rhs, plain_axles = combined_slip_dynamics(sample[:6], sample[6:], params)
        np.testing.assert_array_equal(rhs, plain_rhs)
        np.testing.assert_array_equal(axles, plain_axles)
        counts[index] = int(traces[index, 0, 0])
    @njit
    def batch(kind, repetitions, runtime_params):
        params = runtime_params
        checksum = 0.
        for _ in range(repetitions):
            for index in range(len(samples)):
                sample = samples[index]
                if kind == 0:
                    rhs, axles = combined_slip_dynamics(sample[:6], sample[6:], params)
                    checksum += np.sum(rhs) + np.sum(axles)
                    continue
                c, s = np.cos(sample[6]), np.sin(sample[6])
                for part in range(1, counts[index] + 1):
                    force = traces[index, part]
                    uf, vf, ur, vr, rolling, fzf, fzr, ax = force
                    if kind == 1:
                        residual, axles = _axle_forces(ax, uf, vf, ur, vr, sample[7], c, s, params)
                        checksum += residual + np.sum(axles)
                    elif kind == 2:
                        front = mf61_tire_force(uf, vf, rolling, fzf, params[5], params[8], params[6])
                        rear = mf61_tire_force(ur, vr, rolling, fzr, params[5], params[9], params[6])
                        checksum += sum(front) + sum(rear)
                    else:
                        checksum += np.sum(force)  # replay-loop/checksum floor
        return checksum
    result = {}
    for kind, name in enumerate(("rhs_including_load_solve", "exact_axle_force_sequence",
                                 "exact_mf61_sequence", "replay_loop_floor")):
        started = perf_counter()
        batch(kind, 1, params)
        startup = perf_counter() - started
        values = []
        for _ in range(repeats):
            started = perf_counter()
            checksum = batch(kind, batches, params)
            values.append((perf_counter() - started) / (batches * len(samples)))
        result[name] = dict(seconds_per_rhs_sample=values, median_seconds_per_rhs_sample=float(np.median(values)),
                            checksum=checksum, startup_warmup_seconds=startup)
    iterations = counts - 2
    force_rows = np.concatenate([trace[1:count+1] for trace, count in zip(traces, counts)])
    return dict(kernels=result, samples=len(samples), force_calls=int(counts.sum()),
                mf61_calls=int(2 * counts.sum()),
                zero_load_mf_calls=int(np.count_nonzero(force_rows[:, 5:7] == 0.)),
                iterations_quantiles=np.percentile(iterations, [0, 50, 90, 95, 99, 100]).tolist(),
                iteration_histogram={str(k):int(v) for k,v in zip(*np.unique(iterations, return_counts=True))},
                input_speed_range=[float(samples[:, 3].min()), float(samples[:, 3].max())],
                input_abs_yaw_range=[float(np.abs(samples[:, 5]).min()), float(np.abs(samples[:, 5]).max())],
                exact_force_trace_parity=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maps", nargs="+", default=["circle_map", "Budapest_map"])
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
                    git_status=subprocess.check_output(["git","status","--short"],text=True),
                    command=sys.argv, python=sys.version, platform=platform.platform(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                    timing_scope="serial environment/task only; excludes neural inference, TensorDict collection, GAE, optimization",
                    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [*sorted((ROOT/'src').rglob('*.py')),Path(__file__)]})
    import numba
    import scipy
    metadata['versions'] = dict(numpy=np.__version__, numba=numba.__version__, scipy=scipy.__version__)
    metadata['cpu_model'] = next((line.split(':',1)[1].strip() for line in
                                 Path('/proc/cpuinfo').read_text().splitlines()
                                 if line.startswith('model name')), platform.processor())
    for relative in subprocess.check_output(["git","ls-files","--modified","--others","--exclude-standard"],text=True).splitlines():
        path = ROOT / relative
        if path.is_file() and path.suffix == '.py':
            destination = args.output / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    result = dict(metadata=metadata, workloads={}, collision_guards={})
    suite_started = perf_counter()
    def save():
        result['metadata']['suite_wall_seconds'] = perf_counter() - suite_started
        (args.output/'results.json').write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n')
    for case_index, map_id in enumerate(args.maps):
        print(f"Recording {map_id}: requested {args.steps} steps",flush=True)
        task, drivers, reset, config = build_workload(map_id, args.steps, layout='spaced' if case_index == 0 else 'close')
        try:
            started = perf_counter()
            frames, samples = record_trace(task, drivers, reset, args.steps)
            stats = trajectory_statistics(frames)
            print(f"{map_id}: {stats}",flush=True)
            if len(frames) != args.steps or stats['collision_steps']:
                raise RuntimeError('Representative trace ended early or collided; controlled layer replay requires an uninterrupted collision-free trace')
            np.savez_compressed(args.output/f'{map_id}_trace.npz',
                                physical_actions=np.array([f['controls'] for f in frames]),
                                normalized_actions=np.array([[f['normalized'][aid] for aid in task.possible_agents] for f in frames]),
                                states=np.array([f['state'] for f in frames]),rhs_samples=samples,
                                task_digests=np.array([f['task_digest'] for f in frames]),
                                physics_digests=np.array([f['physics_digest'] for f in frames]))
            workload=dict(config=config,trajectory=stats,recording_seconds=perf_counter()-started,
                          rhs_samples=len(samples),warmup_parity={},runs={layer:[] for layer in LAYERS})
            workload['trajectory_windows'] = [dict(start_step=start,
                                                  **trajectory_statistics(frames[start:start+200]))
                                              for start in range(0,len(frames),200)]
            for layer in LAYERS:
                workload['warmup_parity'][layer] = replay(task,reset,frames,layer,capture=True)
                print(f'{map_id}: {layer} exact replay verified',flush=True)
            for repeat in range(args.repeats):
                for layer in LAYERS[repeat % len(LAYERS):] + LAYERS[:repeat % len(LAYERS)]:
                    row=replay(task,reset,frames,layer)
                    workload['runs'][layer].append(row)
                    print(f"{map_id}: repeat {repeat+1} {layer} {row['steps_per_second']:.1f} steps/s",flush=True)
            for layer in ('environment','task_replay','task_live'):
                validation=replay(task,reset,frames,layer,profile=SerialPhysicsProfile(),capture=True)
                row=replay(task,reset,frames,layer,profile=SerialPhysicsProfile())
                row['exact_physics_parity'] = validation['exact_physics_parity']
                row['exact_task_parity'] = validation['exact_task_parity']
                workload.setdefault('detail',{})[layer]=row
            selected=samples[np.linspace(0,len(samples)-1,min(256,len(samples)),dtype=int)]
            workload['solver_breakdown']=solver_breakdown(selected,task.env.sim.agents[0]._physics._dynamics_params)
            result['workloads'][map_id]=workload
        finally:
            task.close()
        save()
    arena = p1.synthetic_map(args.output)
    for name in ('B4','B5'):
        sim,task,reset,advance,config=p1.build_case(name,arena,ROOT,12)
        disabled=p1.run_rollout(sim,task,reset,advance,12,capture=True)
        measured=p1.run_rollout(sim,task,reset,advance,12,SerialPhysicsProfile(),capture=True)
        assert disabled['digest']==measured['digest']
        result['collision_guards'][name]=dict(config=config,parity=True,runs=[p1.run_rollout(sim,task,reset,advance,12) for _ in range(args.repeats)],
                                             detail=measured)
    save()
    print(args.output/'results.json',flush=True)


if __name__ == '__main__':
    main()
