"""Matched batch/affinity experiments; run inside a SLURM CPU/GPU allocation.

Examples:
  srun --ntasks=1 --cpus-per-task=32 --cpu-bind=verbose,cores \
    python3 scripts/benchmark_collection.py --matrix batches
  srun --ntasks=1 --cpus-per-task=32 --cpu-bind=verbose,cores \
    python3 scripts/benchmark_collection.py --matrix affinity --placement-cpus 16

Affinity uses taskset on the complete trainer process tree, without changing
NUMA memory policy or pinning individual workers. Socket-local and spread use
equal physical-core counts; inherited scheduling retains the full allowed set.
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.cpu_affinity import cpu_placement


def placement_masks(placement, count, minimum):
    """Use one allowed logical CPU per physical core; never widen the allocation."""
    if not placement["cpu_affinity"]:
        raise ValueError("Affinity experiments require Linux sched_getaffinity")
    sockets = {}
    for row in placement["cpu_topology"]:
        if row["socket"] is None or row["core"] is None:
            raise ValueError("Affinity experiments require Linux CPU socket/core topology")
        sockets.setdefault(row["socket"], {}).setdefault(row["core"], row["cpu"])
    pools = [list(cores.values()) for _, cores in sorted(sockets.items())]
    local = max(pools, key=len)
    if count < minimum or count > len(local):
        raise ValueError(f"--placement-cpus must be between {minimum} and {len(local)} "
                         "allocated physical cores on a single socket")
    masks = {"inherited": None, "socket": sorted(local[:count])}
    if len(pools) > 1:
        spread = [pool[index] for index in range(max(map(len, pools)))
                  for pool in pools if index < len(pool)]
        masks["spread"] = sorted(spread[:count])
    return masks


def training_command(args, frames, destination, mask=None):
    scenario = ("ppo_lap_completion_pretrain.yaml" if args.algorithm == "ppo"
                else "mappo_2v2_completion_scratch.yaml")
    target = 4096 if args.algorithm == "ppo" else 2048
    epochs = 4 if args.algorithm == "ppo" else 5
    command = [sys.executable, str(PROJECT_ROOT / "run.py"), "--scenario", f"scenarios/{scenario}",
               "--total-steps", str(args.total_steps), "--seed", "42", "--no-wandb", "--no-render",
               "--torch-threads", str(args.torch_threads), "--output-dir", str(destination),
               "--run-id", destination.name, "--quiet"]
    overrides = [f"experiment.parallel_envs={args.workers}",
                 f"experiment.envs_per_worker={args.envs_per_worker}",
                 f"++algorithm.params.frames_per_batch={frames}",
                 f"algorithm.params.n_steps={target}", "algorithm.params.batch_size=512",
                 f"algorithm.params.n_epochs={epochs}", f"execution.params.device={args.learner_device}",
                 "execution.params.collection_device=cpu", "evaluation.enabled=false",
                 "++experiment.measure_services=true", "++experiment.measure_stages=false",
                 "++experiment.measure_cuda_sync=false", "experiment.measure_mpc=false",
                 "experiment.measure_hybrid=false", "++experiment.profiling_warmup_updates=1",
                 "checkpoint.every_steps=null", "checkpoint.every_updates=null"]
    for override in overrides:
        command.extend(["--set", override])
    if mask is not None:
        command = ["taskset", "--cpu-list", ",".join(map(str, mask)), *command]
    return command


def report_run(destination, target, mask, allowed):
    summary = json.loads((destination / "run_summary.json").read_text())
    actual = summary["environment_steps"]
    batch = summary["perf/last_collection_batch_frames"]
    if not target <= actual < target + batch:
        raise RuntimeError(f"{destination}: invalid collected budget {actual} for target {target}")
    parent = summary["perf/process/cpu_affinity"]
    if (allowed is not None and not set(parent).issubset(allowed)) or (mask is not None and parent != mask):
        raise RuntimeError(f"{destination}: parent affinity does not match the allocated placement")
    worker_masks = {key: value for key, value in summary.items()
                    if key.startswith("perf/workers/") and key.endswith("/cpu_affinity")}
    expected_workers = int(summary["perf/worker_count"])
    expected_processes = expected_workers if summary["perf/environment_slots"] > 1 else 0
    if len(worker_masks) != expected_processes or any(value != parent for value in worker_masks.values()):
        raise RuntimeError(f"{destination}: workers did not inherit the parent affinity")
    boundaries = {stage: sum(value for key, value in summary.items()
                            if key.startswith("perf/boundaries/") and key.endswith(f"/{stage}_seconds"))
                  for stage in ("collection", "aggregation", "update", "weight_sync")}
    print(json.dumps(dict(run=str(destination), decisions=actual, budget_overshoot=actual - target,
                          collection_batches=summary["perf/collection_batches"], actual_batch_frames=batch,
                          decisions_per_second=summary["perf/environment_steps_per_second"],
                          steady_decisions_per_second=summary.get("perf/steady_environment_steps_per_second"),
                          learner_samples=summary["agent_steps"], optimizer_steps=summary["optimizer_steps"],
                          cpu_affinity=parent, worker_affinity_verified=True, **boundaries)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", choices=("batches", "affinity"), required=True)
    parser.add_argument("--algorithm", choices=("ppo", "mappo"), default="ppo")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--envs-per-worker", type=int, default=2)
    parser.add_argument("--frames-per-batch", type=int, default=1024, help="Fixed for the affinity matrix")
    parser.add_argument("--total-steps", type=int, default=262144)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--learner-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--placement-cpus", type=int, default=None,
                        help="Physical cores for socket/spread; default is one thread budget per process")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "phase45")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or creating files")
    args = parser.parse_args()
    if min(args.workers, args.envs_per_worker, args.frames_per_batch, args.total_steps,
           args.repetitions, args.torch_threads) < 1:
        parser.error("Counts and budgets must be positive")
    placement = cpu_placement()
    print("Inherited placement: " + json.dumps(placement), flush=True)
    masks = {"inherited": None}
    if args.matrix == "affinity":
        if shutil.which("taskset") is None:
            parser.error("Affinity experiments require taskset")
        try:
            minimum = (args.workers + 1) * args.torch_threads
            count = minimum if args.placement_cpus is None else args.placement_cpus
            masks = placement_masks(placement, count, minimum)
        except ValueError as exc:
            parser.error(str(exc))
    frames = (512, 1024, 2048) if args.matrix == "batches" else (args.frames_per_batch,)
    for repeat in range(1, args.repetitions + 1):
        # Rotate order to reduce bias from node load/warm caches between repetitions.
        candidates = [(size, name, mask) for size in frames for name, mask in masks.items()]
        offset = (repeat - 1) % len(candidates)
        for size, name, mask in candidates[offset:] + candidates[:offset]:
            destination = args.output_root.resolve() / args.algorithm / args.matrix / (
                f"w{args.workers}e{args.envs_per_worker}_f{size}_{name}_r{repeat}")
            command = training_command(args, size, destination, mask)
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            if destination.exists():
                parser.error(f"Output already exists: {destination}; use another --output-root")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            report_run(destination, args.total_steps, mask, placement["cpu_affinity"])


if __name__ == "__main__":
    main()
