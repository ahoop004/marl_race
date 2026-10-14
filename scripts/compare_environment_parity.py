"""Compare isolated historical/current environments without training or source edits.

Example: python3 scripts/compare_environment_parity.py --baseline-repo ../F110_MARL
The baseline is exported from --baseline-commit, never imported from its checkout.
JSON traces and resolved configs are retained under --output (default /tmp).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

SCENARIO = "scenarios/ppo_lap_completion_pretrain.yaml"
POSE = [60.440993, 118.69, float(np.pi)]
INFO_FIELDS = ("centerline", "track_limits", "lap_count", "lap_crossed",
               "race_completed", "lap_start_step", "lap_time_steps", "collision",
               "collision_event", "boundary_event", "terminal_reason", "status",
               "terminal_step", "finish_position")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def worker(args):
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from core.scenario import load_and_expand_scenario
    from wrappers.actions.composer import ActionComposer
    from wrappers.rewards.composer import RewardComposer

    cfg = (load_and_expand_scenario(SCENARIO, overrides=[f"reward={args.reward_preset}"])
           if args.side == "current" and args.reward_preset
           else load_and_expand_scenario(SCENARIO))
    original = serial(cfg)
    cfg["environment"]["friction"]["train"] = {"mode": "fixed", "mu": 1.0489}
    if args.side == "baseline":
        from core.setup import create_training_setup, build_obs_composer, build_reward_composer
        from training.reward_context import build_reward_context
        env, _, _ = create_training_setup(cfg, scenario_dir=root / "scenarios")
        obs = build_obs_composer(cfg["agents"]["car_0"], cfg["environment"], root / "scenarios")
        rewards = build_reward_composer(cfg["agents"]["car_0"], root / "scenarios")
        space = env.action_spaces["car_0"]
        actions = ActionComposer.from_config(space.low, space.high,
            cfg["agents"]["car_0"]["action_constraints"], decision_dt=env.timestep)
    else:
        import torch
        from adapters.native_torchrl import NativeRaceTorchRLEnv
        native = NativeRaceTorchRLEnv.from_scenario(cfg, scenario_dir=root / "scenarios")
        env = native.task.env
        obs = native.task.obs_composers["car_0"]
        rewards = native.task.reward_composers["car_0"]

    # The existing scanner accepts rng=None: geometric scans with no white noise.
    # Explicit poses/velocities and fixed grip remove spawn/friction RNG dependence.
    car = env.sim.agents[0]
    scan = car.scan_simulator.scan
    # Vehicle.reset recreates scan_rng, so disable noise at the scanner call.
    car.scan_simulator.scan = lambda pose, rng: scan(pose, None)
    result = {"configuration": original, "observation_contract": obs.contract,
              "runtime": {"params": env.params, "max_steps": env.max_steps,
                          "integrator": str(car.integrator), "fov": car.fov,
                          "lidar_offset": car.lidar_dist}, "traces": {}}
    try:
        for name in ("straight", "alternating", "wall"):
            options = {"poses": [POSE], "velocities": [0.0]}
            if args.side == "baseline":
                raw, infos = env.reset(options=options)
                for composer in (obs, rewards, actions):
                    composer.reset()
                vector = obs.wrap(raw["car_0"], infos["car_0"])
            else:
                td = native.reset(options=options)
                raw, infos = native.snapshot.raw_observations, native.snapshot.infos
                vector = td["agents", "observation"][0].numpy()

            def record(tick, normalized=None, physical=None, reward=0.0,
                       components=None, terminated=False, truncated=False):
                return serial({"decision": tick, "normalized_action": normalized,
                    "physical_action": physical, "physics_state": car.physics_state.copy(),
                    "control_reference": car.control_reference.copy(),
                    "pose": raw["car_0"]["pose"], "velocity": raw["car_0"]["velocity"],
                    "lidar": raw["car_0"]["scans"], "observation": vector,
                    "info": {k: infos["car_0"].get(k, False if k.endswith("event") else None)
                             for k in INFO_FIELDS},
                    "lap_counts": env.lap_counts.copy(), "collision_index": env.sim.collision_idx.copy(),
                    "reward_components": components or {}, "reward": reward,
                    "terminated": terminated, "truncated": truncated,
                    "episode_done": env.episode_done})

            trace = [record(0)]
            for tick in range(1, args.decisions + 1):
                steer = 0.0 if name == "straight" else (
                    (0.5 if ((tick - 1) // 20) % 2 == 0 else -0.5)
                    if name == "alternating" else 1.0)
                normalized = np.array([steer, 0.6], dtype=np.float32)
                if args.side == "baseline":
                    physical = actions.process(normalized)
                    raw, _, terms, truncs, infos = env.step({"car_0": physical})
                    terminated, truncated = terms["car_0"], truncs["car_0"]
                    context = build_reward_context(env=env, agent_id="car_0", info_dict=infos,
                        obs_dict=raw, actions={"car_0": physical}, global_state=env.get_global_state())
                    context.update(obs=vector, next_obs=raw["car_0"], info=infos["car_0"],
                        done=terminated or truncated, terminated=terminated, truncated=truncated,
                        action=normalized, timestep=env.timestep)
                    reward, components = rewards.compute(context)
                    vector = obs.wrap(raw["car_0"], infos["car_0"])
                else:
                    td["agents", "action"] = torch.from_numpy(normalized[None].copy())
                    td = native.step(td)["next"]
                    step = native.last_step
                    decision = step.decisions["car_0"]
                    physical = decision.action_physical
                    raw, infos = step.after.raw_observations, step.after.infos
                    vector = td["agents", "observation"][0].numpy()
                    reward = float(td["agents", "reward"].item())
                    components = decision.reward_components
                    terminated, truncated = bool(td["terminated"].item()), bool(td["truncated"].item())
                    assert terminated == decision.terminated and truncated == decision.truncated
                    np.testing.assert_array_equal(vector, decision.next_observation)
                trace.append(record(tick, normalized, physical, reward, components, terminated, truncated))
                if env.episode_done:
                    break
            result["traces"][name] = trace

        # Focused reward contexts isolate changed event predicates; no physics mutation.
        result["reward_probes"] = {}
        probes = {
            "forward_progress": {"centerline": {"progress_delta": .01}},
            "track_boundary": {"track_limits": {"exceeded": True}},
            "boundary_event_only": {"boundary_event": True},
            "terminating_collision": {"terminal_reason": "collision", "collision": True, "collision_event": True},
            "clean_finish": {"lap_crossed": True, "race_completed": True, "terminal_reason": "race_complete"},
            "nonterminal_collision": {"collision": True, "collision_event": True,
                                      "terminal_reason": None, "centerline": {"progress_delta": .001}},
            "finish_and_boundary": {"lap_crossed": True, "race_completed": True, "terminal_reason": "race_complete",
                                    "boundary_event": True, "track_limits": {"exceeded": True}},
            "finish_and_collision": {"lap_crossed": True, "race_completed": True, "terminal_reason": "race_complete",
                                     "collision": True, "collision_event": True},
            "finish_and_collision_event_only": {"lap_crossed": True, "race_completed": True,
                                                 "collision": False, "collision_event": True},
            "finish_with_collision_reason": {"lap_crossed": True, "race_completed": True,
                                             "terminal_reason": "collision", "collision": False},
            "finish_with_timeout_reason": {"lap_crossed": True, "race_completed": True,
                                           "terminal_reason": "time_limit"},
            "timeout": {"terminal_reason": "time_limit", "TimeLimit.truncated": True},
            "terminal_step_time": {},
        }
        for name, info in probes.items():
            info = {"track_limits": {"exceeded": False},
                    "centerline": {"progress_delta": 0.0}, **info}
            composer = RewardComposer.from_config(cfg["agents"]["car_0"]["reward"])
            result["reward_probes"][name] = composer.compute({"info": info,
                "done": name not in ("forward_progress", "track_boundary", "boundary_event_only", "nonterminal_collision"),
                "terminated": name not in ("timeout", "finish_with_timeout_reason", "forward_progress",
                                           "track_boundary", "boundary_event_only", "nonterminal_collision"),
                "truncated": name in ("timeout", "finish_with_timeout_reason"), "timestep": .05})
        from env.centerline_state import LapTracker
        from env.collision_state import RaceLifecycle
        finish = env._finish_line_data
        middle = (np.asarray(finish["start"]) + np.asarray(finish["end"])) / 2
        direction = np.asarray(finish["direction"])
        tangent = np.asarray(finish["segment_unit"])
        before, after = middle - direction, middle + direction
        # A geometric circuit around the finish endpoint avoids reverse debt.
        # These tracker-only probes do not claim a physically driven clean lap.
        bypass = tangent * (np.linalg.norm(finish["segment"]) + 2)
        result["lap_probes"] = {}
        for event in ("clean", "boundary", "collision"):
            lifecycle = RaceLifecycle(("car_0",), target_laps=1)
            tracker = LapTracker(("car_0",), finish, lifecycle, count_initial_crossing_as_lap=False)
            tracker.reset(np.array([before[0]]), np.array([before[1]]))
            crossings = []
            for tick, point in enumerate((after, after+bypass, before+bypass, before, after), 1):
                crossings.append(tracker.update(np.array([point[0]]), np.array([point[1]]),
                    np.array([1.0]), np.array([0.0]), step=tick)["car_0"])
            if event == "collision":
                lifecycle.record_collision("car_0", step=tick)
            elif event == "boundary":
                lifecycle.record_track_boundary("car_0", step=tick)
            state = lifecycle.records["car_0"]
            result["lap_probes"][event] = {"crossings": crossings, "lap_count": state.lap_count,
                "status": state.status.value, "terminal_reason": state.terminal_reason.value}
    finally:
        (env if args.side == "baseline" else native).close()
    args.output.write_text(json.dumps(serial(result), indent=2))


def compare(a, b, path="", atol=1e-6, rtol=1e-5):
    differences = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            differences.extend(compare(a.get(key), b.get(key), f"{path}.{key}", atol, rtol))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (left, right) in enumerate(zip(a, b)):
            differences.extend(compare(left, right, f"{path}[{i}]", atol, rtol))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not np.isclose(a, b, atol=atol, rtol=rtol):
            differences.append({"field": path, "baseline": a, "current": b,
                "absolute_error": abs(a-b), "relative_error": abs(a-b)/max(abs(a), 1e-15)})
    elif a != b:
        differences.append({"field": path, "baseline": a, "current": b})
    return differences


def main(args):
    import tempfile
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(args.baseline_repo), "rev-parse",
                                      args.baseline_commit], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="f110-parity-") as directory:
        baseline = Path(directory)
        archive = args.output / "baseline.tar"
        with archive.open("wb") as out:
            subprocess.run(["git", "-C", str(args.baseline_repo), "archive", commit], stdout=out, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(baseline, filter="data")
        archive.unlink()
        for side, repo in (("baseline", baseline), ("current", root)):
            command = [sys.executable, str(Path(__file__).resolve()), "--side", side,
                "--root", str(repo), "--output", str(args.output / f"{side}.json"),
                "--decisions", str(args.decisions)]
            if args.reward_preset:
                command.extend(["--reward-preset", args.reward_preset])
            subprocess.run(command, check=True,
                env={**os.environ, "PYGLET_HEADLESS": "true", "PYTHONDONTWRITEBYTECODE": "1"})
        old, new = [json.loads((args.output / f"{side}.json").read_text()) for side in ("baseline", "current")]
        assets = {}
        for path in sorted((baseline / "maps/circle_map").rglob("*")):
            if path.is_file():
                relative = path.relative_to(baseline)
                hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, root / relative)]
                assets[str(relative)] = {"baseline": hashes[0], "current": hashes[1], "equal": hashes[0] == hashes[1]}
    report = {"baseline_commit": commit, "current_commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "initial_pose": POSE,
        "reward_preset": args.reward_preset or "solo_completion",
        "initial_state": "zero vx/vy/yaw-rate/steering/wheel-speed/references", "friction": 1.0489,
        "sensor_noise": "disabled with existing scanner rng=None", "assets": assets, "scenarios": {}}
    for name, left in old["traces"].items():
        right = new["traces"][name]
        first, maxima = None, {}
        for a, b in zip(left, right):
            differences = compare(a, b)
            if differences and first is None:
                first = {"decision": a["decision"], "differences": differences}
            for key in ("physics_state", "control_reference", "pose", "velocity", "lidar", "observation", "reward"):
                delta = float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key]))))
                maxima[key] = max(maxima.get(key, 0.0), delta)
        report["scenarios"][name] = {"first_divergence": first, "max_absolute_errors": maxima,
            "baseline_decisions": len(left)-1, "current_decisions": len(right)-1,
            "baseline_outcome": left[-1]["info"]["terminal_reason"],
            "current_outcome": right[-1]["info"]["terminal_reason"],
            "baseline_return": sum(t["reward"] for t in left), "current_return": sum(t["reward"] for t in right)}
    report["reward_probe_differences"] = compare(old["reward_probes"], new["reward_probes"])
    report["reward_probes"] = {"baseline": old["reward_probes"], "current": new["reward_probes"]}
    report["lap_probes"] = new["lap_probes"]
    report["lap_probe_differences"] = compare(old["lap_probes"], new["lap_probes"])
    report["configuration_differences"] = compare(old["configuration"]["environment"], new["configuration"]["environment"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "assets"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, default=Path("/home/aaron/F110_MARL"))
    parser.add_argument("--baseline-commit", default="71a3e0f")
    parser.add_argument("--decisions", type=int, default=1000)
    parser.add_argument("--reward-preset", help="Current Hydra reward preset; baseline remains historical")
    parser.add_argument("--output", type=Path, default=Path("/tmp/f110-environment-parity"))
    parser.add_argument("--side", choices=("baseline", "current"))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    worker(args) if args.side else main(args)
