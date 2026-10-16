#!/usr/bin/env python3
"""Initial independent PPO/MAPPO Ray Tune smoke tests (Ray 2.59; no Ray Train/RLlib).

Run only inside the authenticated, Slurm-allocated Ray cluster. Ray reserves
one GPU and six logical CPUs per trial. The trainer remains a subprocess of
the Tune worker, retaining the existing TorchRL PPO/MAPPO implementation.
"""

import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import ray
from ray import tune

REPO = Path.home() / "marl_race"
SCENARIOS = {
    "ppo": "scenarios/ppo_lap_completion_pretrain.yaml",
    "mappo": "scenarios/mappo_2v2_completion_scratch.yaml",
}


def latest_update(path: Path) -> dict | None:
    """Read the latest fully flushed update row; tolerate a concurrent CSV rewrite."""
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (FileNotFoundError, OSError, csv.Error):
        return None
    return rows[-1] if rows else None


def numeric(row: dict, key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in (None, "") else 0


def trial(config: dict) -> None:
    """One Ray Tune worker -> one existing Python trainer -> one physical GPU."""
    from ray import tune

    label = config["algorithm"]
    context = tune.get_context()
    trial_id = context.get_trial_id()
    root = Path(os.environ["RAY_TUNE_JOB_ROOT"])
    out = root / "trial_runs" / f"{label}_{trial_id}"
    out.mkdir(parents=True, exist_ok=False)

    allocated = ray.get_gpu_ids()
    if len(allocated) != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(f"Expected 1 Ray GPU, got {allocated}, CVD={os.getenv('CUDA_VISIBLE_DEVICES')!r}")

    run_id = f"ray_tune_{label}_{trial_id}"
    cmd = [
        sys.executable, "run.py",
        "--scenario", SCENARIOS[label],
        "--output-dir", str(out),
        "--run-id", run_id,
        "--total-steps", "4096",
        "--seed", "42",
        "--torch-threads", "1",
        "--set", "experiment.parallel_envs=2",
        "--set", "experiment.envs_per_worker=1",
        "--set", "experiment.distributed=false",
        "--set", "algorithm.params.n_steps=1024",
        "--set", "algorithm.params.frames_per_batch=256",
        "--set", "algorithm.params.batch_size=128",
        "--set", "algorithm.params.n_epochs=2",
        "--set", "checkpoint.every_steps=2048",
        "--set", "evaluation.enabled=false",
        "--set", "execution.params.device=cuda",
        "--set", "execution.params.collection_device=cpu",
        "--no-wandb", "--no-render",
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "WEIGHT_SYNC_TIMEOUT": "900",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYGLET_HEADLESS": "true",
        "F110_SUMMARY_EVERY": "10",
    })
    metadata = {
        "algorithm": label,
        "hostname": socket.gethostname(),
        "ray_gpu_ids": [str(x) for x in allocated],
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "command": cmd,
        "git_sha": os.environ.get("RAY_TUNE_GIT_SHA"),
        "training_output_dir": str(out),
    }
    (out / "ray_trial.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Starting {label} on {metadata['hostname']} GPU={allocated} output={out}", flush=True)

    process = None
    last_reported_update = 0
    update_csv = out / "update_metrics.csv"
    timeout_seconds = int(os.environ.get("RAY_TUNE_TRIAL_TIMEOUT", "1200"))
    started = time.monotonic()
    try:
        with (out / "train.log").open("w", encoding="utf-8") as logfile:
            process = subprocess.Popen(
                cmd, cwd=REPO, env=env, stdout=logfile,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            while process.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError(f"{label} exceeded {timeout_seconds}s; see {out / 'train.log'}")
                row = latest_update(update_csv)
                if row is not None:
                    updates = numeric(row, "train/updates")
                    if updates > last_reported_update:
                        last_reported_update = updates
                        tune.report({
                            "algorithm": label,
                            "hostname": metadata["hostname"],
                            "updates": updates,
                            "environment_steps": numeric(row, "train/environment_steps"),
                            "learner_samples": numeric(row, "train/agent_steps"),
                            "optimizer_steps": numeric(row, "train/optimizer_steps_total"),
                            "training_output_dir": str(out),
                            "status": "running",
                        })
                time.sleep(3)
            returncode = process.wait()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    if returncode != 0:
        raise RuntimeError(f"{label} trainer exited {returncode}; inspect {out / 'train.log'}")
    row = latest_update(update_csv)
    if row is None or numeric(row, "train/updates") < 1:
        raise RuntimeError(f"{label}: no optimizer update recorded in {update_csv}")
    if numeric(row, "train/environment_steps") < 4096:
        raise RuntimeError(f"{label}: training budget not reached, see {update_csv}")
    if not (out / "latest.pt").is_file():
        raise RuntimeError(f"{label}: missing latest.pt in {out}")

    tune.report({
        "algorithm": label,
        "hostname": metadata["hostname"],
        "updates": numeric(row, "train/updates"),
        "environment_steps": numeric(row, "train/environment_steps"),
        "learner_samples": numeric(row, "train/agent_steps"),
        "optimizer_steps": numeric(row, "train/optimizer_steps_total"),
        "training_output_dir": str(out),
        "checkpoint_path": str(out / "latest.pt"),
        "status": "completed",
    })


def main() -> None:
    mode = os.environ.get("TRIAL_MODE", "ppo")
    if mode not in {"ppo", "both"}:
        raise ValueError("TRIAL_MODE must be ppo or both")
    algorithms = ["ppo"] if mode == "ppo" else ["ppo", "mappo"]
    root = Path(os.environ["RAY_TUNE_JOB_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            resources = ray.cluster_resources()
            if resources.get("GPU", 0) >= 2 and len([n for n in ray.nodes() if n["Alive"]]) == 2:
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"Two-node Ray cluster not ready: {ray.cluster_resources()}")

        trainable = tune.with_resources(trial, resources={"cpu": 6, "gpu": 1})
        tuner = tune.Tuner(
            trainable,
            param_space={"algorithm": tune.grid_search(algorithms)},
            tune_config=tune.TuneConfig(max_concurrent_trials=len(algorithms)),
            run_config=tune.RunConfig(
                name="independent_trials",
                storage_path=str(root / "tune_results"),
                verbose=1,
            ),
        )
        result_grid = tuner.fit()
        results = []
        for result in result_grid:
            metrics = result.metrics or {}
            results.append({
                "algorithm": metrics.get("algorithm"),
                "hostname": metrics.get("hostname"),
                "status": metrics.get("status"),
                "environment_steps": metrics.get("environment_steps"),
                "training_output_dir": metrics.get("training_output_dir"),
                "tune_result_path": str(result.path),
                "error": str(result.error) if result.error else None,
            })
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
        if len(results) != len(algorithms) or any(r["error"] or r["status"] != "completed" for r in results):
            raise RuntimeError("One or more Ray Tune trials failed")
        if mode == "both" and len({r["hostname"] for r in results}) != 2:
            raise RuntimeError("Two trials did not execute on separate nodes")
        print("ALL RAY TUNE TRAINING CHECKS PASSED", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
