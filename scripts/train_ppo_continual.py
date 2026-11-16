"""Run the full and rank-4 continual PPO conditions as two independent GPU tasks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("full_finetune", "ppo_lora")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--steps-per-task", type=int)
    args = parser.parse_args()
    rank, size = int(os.environ.get("SLURM_PROCID", "0")), int(os.environ.get("SLURM_NTASKS", "1"))
    if size != 2 or not 0 <= rank < size:
        parser.error("Exactly two Slurm tasks are required (full fine-tuning and LoRA)")
    if min(args.workers, args.envs_per_worker) < 1 or (
            args.steps_per_task is not None and args.steps_per_task < 1):
        parser.error("Workers, environment counts and step budgets must be positive")
    affinity = sorted(os.sched_getaffinity(0))
    if args.workers >= len(affinity):
        parser.error("Allocate more CPUs/task than collector workers, leaving a CPU for the learner")
    root = args.root.expanduser().resolve()
    source = root / "source.pt"
    if not source.is_file():
        parser.error(f"Missing immutable source snapshot: {source}")
    condition = CONDITIONS[rank]
    out = root / condition
    out.mkdir(exist_ok=True)
    environment = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    with (out / ".training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen, latest = out / "launch_config.yaml", out / "latest.pt"
        if not args.resume and (frozen.exists() or latest.exists()):
            raise RuntimeError(f"Existing experiment {out}; use --resume")
        if args.resume and not frozen.exists() and any(p.name != ".training.lock" for p in out.iterdir()):
            raise RuntimeError(f"Missing launch_config.yaml in {out}; start with a new root")
        if not frozen.exists():
            resolve = [sys.executable, str(REPO / "run.py"), "--scenario", "scenarios/ppo_continual.yaml",
                "--checkpoint", str(source), "--seed", str(args.seed), "--no-render", "--resolve-config",
                "--set", f"continual={args.profile}", "--set", f"adaptation={condition}",
                "--set", f"experiment.parallel_envs={args.workers}",
                "--set", f"experiment.envs_per_worker={args.envs_per_worker}",
                "--set", f"wandb.group={root.name}", "--set", f"wandb.name={root.name}_{condition}",
                "--set", f"++wandb.id='{uuid.uuid4().hex}'"]
            if args.steps_per_task is not None:
                resolve += ["--set", f"continual.steps_per_task={args.steps_per_task}"]
            resolved = subprocess.run(resolve, cwd=REPO, env=environment, check=True,
                                      stdout=subprocess.PIPE, text=True)
            temporary = frozen.with_suffix(".tmp")
            temporary.write_text(resolved.stdout)
            temporary.replace(frozen)
        config = yaml.safe_load(frozen.read_text())
        if config["experiment"]["parallel_envs"] >= len(affinity):
            raise RuntimeError("Allocate more CPUs/task than the frozen collector worker count")
        if config["adaptation"]["checkpoint"] != str(source):
            raise RuntimeError("Frozen continual experiment source changed")
        if args.resume and (out / "continual_results.json").exists():
            results = json.loads((out / "continual_results.json").read_text())
            if results.get("complete") and (out / "final.pt").is_file():
                print(f"{condition}: already completed", flush=True)
                return
        seed = config["experiment"]["seed"]
        command = [sys.executable, str(REPO / "run.py"), "--scenario", str(frozen),
            "--output-dir", str(out), "--run-id", f"{root.name}_{condition}_s{seed}", "--no-render"]
        command += ["--resume", str(latest)] if args.resume and latest.exists() else ["--checkpoint", str(source)]
        attempt = os.environ.get("SLURM_JOB_ID", "local")
        (out / f"command_{attempt}.json").write_text(json.dumps(dict(condition=condition,
            command=command, cpu_affinity=affinity, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            node=os.environ.get("SLURMD_NODENAME")), indent=2) + "\n")
        print(f"Task {rank}: {condition}; output={out}", flush=True)
        with (out / "train.log").open("a") as log:
            result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{condition}: exit {result.returncode}; see {out / 'train.log'}")
        results = json.loads((out / "continual_results.json").read_text())
        if not results["complete"] or not (out / "final.pt").exists():
            raise RuntimeError(f"{condition}: stopped before all five stages completed; use --resume")


if __name__ == "__main__":
    main()
