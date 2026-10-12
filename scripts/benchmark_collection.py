"""Matched final, batch, affinity, and inference experiments on HPC.

Examples:
  srun --ntasks=1 --cpus-per-task=32 --cpu-bind=verbose,cores \
    python3 scripts/benchmark_collection.py --matrix final --algorithm ppo
  srun --ntasks=1 --cpus-per-task=32 --cpu-bind=verbose,cores \
    python3 scripts/benchmark_collection.py --matrix batches
  srun --ntasks=1 --cpus-per-task=32 --cpu-bind=verbose,cores \
    python3 scripts/benchmark_collection.py --matrix affinity --placement-cpus 16

Affinity uses taskset on the complete trainer process tree, without changing
NUMA memory policy or pinning individual workers. Socket-local and spread use
equal physical-core counts; inherited scheduling retains the full allowed set.
Use --source-root for a baseline checkout, with a separate --output-root. Older
checkouts may lack affinity/batch instrumentation; absent measurements stay null.
Inference/worker timings overlap collection; GAE/optimization overlap update.
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.cpu_affinity import cpu_placement


def run_training(command, destination, profile, source_root=PROJECT_ROOT):
    if not profile or shutil.which("nvidia-smi") is None:
        subprocess.run(command, cwd=source_root, check=True)
        return
    # Device-wide samples include startup/saving and other GPU users. Keep this
    # diagnostic in separate profiled runs, outside the throughput comparison.
    with tempfile.TemporaryFile(mode="w+") as samples:
        sampler = subprocess.Popen([
            "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits", "--loop-ms=500"],
            stdout=samples, stderr=subprocess.DEVNULL)
        try:
            subprocess.run(command, cwd=source_root, check=True)
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        summary_path = destination / "run_summary.json"
        summary = json.loads(summary_path.read_text())
        rows = []
        samples.seek(0)
        for line in samples:
            try:
                index, utilization, memory = map(float, line.strip().split(","))
            except ValueError:
                continue
            if index == summary.get("perf/cuda_device_index"):
                rows.append((utilization, memory))
        if rows:
            prefix = "perf/cuda_device_sampling/"
            summary.update({prefix + "scope": "device-wide, whole run including startup and saving",
                            prefix + "interval_seconds": .5, prefix + "samples": len(rows),
                            prefix + "mean_utilization_percent": sum(row[0] for row in rows) / len(rows),
                            prefix + "max_utilization_percent": max(row[0] for row in rows),
                            prefix + "peak_memory_used_bytes": int(max(row[1] for row in rows) * 1024**2)})
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")


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


def training_command(args, frames, destination, mask=None, *, backend="process", collection_device="cpu",
                     workers=None, envs_per_worker=None):
    scenario = ("ppo_lap_completion_pretrain.yaml" if args.algorithm == "ppo"
                else "mappo_2v2_completion_scratch.yaml")
    target = 4096 if args.algorithm == "ppo" else 2048
    epochs = 4 if args.algorithm == "ppo" else 5
    source_root = getattr(args, "source_root", PROJECT_ROOT)
    workers = args.workers if workers is None else workers
    envs_per_worker = args.envs_per_worker if envs_per_worker is None else envs_per_worker
    command = [sys.executable, str(source_root / "run.py"), "--scenario", f"scenarios/{scenario}",
               "--total-steps", str(args.total_steps), "--seed", "42", "--no-wandb", "--no-render",
               "--torch-threads", str(args.torch_threads), "--output-dir", str(destination),
               "--run-id", destination.name, "--quiet"]
    overrides = [f"experiment.parallel_envs={workers}",
                 f"experiment.envs_per_worker={envs_per_worker}",
                 f"++algorithm.params.frames_per_batch={frames}",
                 f"algorithm.params.n_steps={target}", "algorithm.params.batch_size=512",
                 f"algorithm.params.n_epochs={epochs}", f"execution.params.device={args.learner_device}",
                 f"execution.params.collection_device={collection_device}", "evaluation.enabled=false",
                 "++experiment.measure_services=true",
                 f"++experiment.measure_stages={str(getattr(args, 'profile', False)).lower()}",
                 f"++experiment.measure_cuda_sync={str(getattr(args, 'profile', False)).lower()}",
                 "experiment.measure_mpc=false",
                 "experiment.measure_hybrid=false", "++experiment.profiling_warmup_updates=1",
                 "checkpoint.every_steps=null", "checkpoint.every_updates=null"]
    if backend != "process":
        overrides.append(f"++experiment.collection_backend={backend}")
    for override in overrides:
        command.extend(["--set", override])
    if mask is not None:
        command = ["taskset", "--cpu-list", ",".join(map(str, mask)), *command]
    return command


def report_run(destination, target, mask, allowed, *, backend="process", collection_device="cpu",
               learner_device=None):
    summary = json.loads((destination / "run_summary.json").read_text())
    snapshot = json.loads((destination / "config_snapshot.json").read_text())
    config = snapshot["config"]
    execution = snapshot["provenance"]["execution"]
    actual = summary["environment_steps"]
    # The pre-optimization grouped collector did not export batch/placement data.
    # Its fixed synchronous batch size can be derived from the saved configuration.
    slots = int(summary["perf/environment_slots"])
    frames = config["algorithm"]["params"]["frames_per_batch"]
    batch = summary.get("perf/last_collection_batch_frames", ((frames + slots - 1) // slots) * slots)
    for name, expected in (("collection_backend", backend), ("collection_device", collection_device),
                           ("learner_device", learner_device)):
        measured = summary.get(f"perf/{name}", execution.get(name, "process" if name == "collection_backend" else None))
        if expected is not None and measured != expected:
            raise RuntimeError(f"{destination}: {name} did not match requested {expected}")
    if config["experiment"]["total_steps"] != target:
        raise RuntimeError(f"{destination}: saved decision target did not match requested {target}")
    if not target <= actual < target + batch:
        raise RuntimeError(f"{destination}: invalid collected budget {actual} for target {target}")
    parent = summary.get("perf/process/cpu_affinity")
    if parent is not None and ((allowed is not None and not set(parent).issubset(allowed))
                               or (mask is not None and parent != mask)):
        raise RuntimeError(f"{destination}: parent affinity does not match the allocated placement")
    worker_masks = {key: value for key, value in summary.items()
                    if key.startswith("perf/workers/") and key.endswith("/cpu_affinity")}
    expected_workers = int(summary["perf/worker_count"])
    expected_processes = expected_workers if summary["perf/environment_slots"] > 1 or backend == "centralized" else 0
    affinity_verified = None
    if parent is not None:
        if len(worker_masks) != expected_processes or any(value != parent for value in worker_masks.values()):
            raise RuntimeError(f"{destination}: workers did not inherit the parent affinity")
        affinity_verified = True
    elif worker_masks:
        raise RuntimeError(f"{destination}: workers did not inherit the parent affinity")
    worker_pids = {value for key, value in summary.items()
                   if key.startswith("perf/workers/") and key.endswith("/pid")}
    if worker_pids and len(worker_pids) != expected_processes:
        raise RuntimeError(f"{destination}: worker process count did not match the configured layout")
    boundaries = {stage: sum(value for key, value in summary.items()
                            if key.startswith("perf/boundaries/") and key.endswith(f"/{stage}_seconds"))
                  for stage in ("collection", "aggregation", "update", "weight_sync")}
    report = dict(run=str(destination), decisions=actual, budget_overshoot=actual - target,
                          collection_batches=summary.get("perf/collection_batches"), actual_batch_frames=batch,
                          batch_frames_measured="perf/last_collection_batch_frames" in summary,
                          decisions_per_second=summary["perf/environment_steps_per_second"],
                          steady_decisions_per_second=summary.get("perf/steady_environment_steps_per_second"),
                          learner_samples_per_second=summary["perf/learner_samples_per_second"],
                          physics_steps_per_second=summary["perf/physics_steps_per_second"],
                          physics_steps=summary["physics_steps"], completed_episodes=summary["completed_episodes"],
                          learner_samples=summary["agent_steps"], optimizer_steps=summary["optimizer_steps"],
                          backend=backend, collection_device=collection_device,
                          cpu_affinity=parent, worker_affinity_verified=affinity_verified,
                          expected_worker_processes=expected_processes,
                          actual_worker_processes=len(worker_pids) if worker_pids else (0 if not expected_processes else None),
                          source_commit=snapshot["provenance"].get("git", {}).get("commit"),
                          source_dirty=snapshot["provenance"].get("git", {}).get("dirty"),
                          worker_startup_seconds=summary.get("perf/boundaries/initialization/worker_startup_seconds"),
                          parent_peak_rss_bytes=summary.get("perf/process/peak_rss_bytes"),
                          worker_wait_seconds=sum(value for key, value in summary.items()
                              if key.startswith("perf/boundaries/") and key.endswith("/worker_wait_seconds")),
                          worker_wait_overlaps_collection=True, **boundaries,
                          profiling={key: value for key, value in summary.items()
                                     if key.startswith(("perf/inference/", "perf/cuda_collection/",
                                                        "perf/cuda_device_sampling/"))})
    print(json.dumps(report), flush=True)
    return report


def parse_layout(value):
    try:
        layout = tuple(int(part) for part in value.split("x"))
    except ValueError:
        layout = ()
    if len(layout) != 3 or min(layout) < 1:
        raise argparse.ArgumentTypeError("Layouts must be positive WORKERSxENVS_PER_WORKERxFRAMES, e.g. 8x2x1024")
    return layout


def matrix_candidates(args, masks):
    if args.matrix == "final":
        layouts = args.layout or ([(4, 2, 1024), (8, 1, 1024), (8, 2, 1024), (8, 4, 1024),
                                  (16, 2, 1024), (8, 2, 512), (8, 2, 2048), (8, 4, 2048)]
                                 if args.algorithm == "ppo" else
                                 [(4, 2, 1024), (8, 2, 1024), (8, 4, 1024)])
        return [(workers, envs, size, "inherited", None, "process", "cpu")
                for workers, envs, size in dict.fromkeys(layouts)]
    if args.matrix == "inference":
        return [(args.workers, args.envs_per_worker, args.frames_per_batch, name, None, backend, device)
                for name, backend, device in (("grouped_cpu", "process", "cpu"),
                    ("central_cpu", "centralized", "cpu"), ("central_cuda", "centralized", "cuda"))]
    frames = (512, 1024, 2048) if args.matrix == "batches" else (args.frames_per_batch,)
    return [(args.workers, args.envs_per_worker, size, name, mask, "process", "cpu")
            for size in frames for name, mask in masks.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", choices=("final", "batches", "affinity", "inference"), required=True)
    parser.add_argument("--algorithm", choices=("ppo", "mappo"), default="ppo")
    parser.add_argument("--workers", type=int, default=8,
                        help="Workers for batches/affinity/inference; use --layout for final")
    parser.add_argument("--envs-per-worker", type=int, default=2,
                        help="Envs per worker for batches/affinity/inference; use --layout for final")
    parser.add_argument("--frames-per-batch", type=int, default=1024, help="Fixed for affinity/inference matrices")
    parser.add_argument("--layout", type=parse_layout, action="append",
                        help="Select a final-matrix shortlist; repeat for multiple WORKERSxENVSxFRAMES layouts")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT,
                        help="Training checkout, e.g. a baseline worktree; the same runner drives both revisions")
    parser.add_argument("--total-steps", type=int, default=262144)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--learner-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--placement-cpus", type=int, default=None,
                        help="Physical cores for socket/spread; default is one thread budget per process")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "hpc_final")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or creating files")
    parser.add_argument("--profile", action="store_true",
                        help="Enable detailed inference/transfer timing; use separate runs for throughput")
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    if not (args.source_root / "run.py").is_file():
        parser.error("--source-root must contain run.py")
    if args.layout and args.matrix != "final":
        parser.error("--layout is only supported with --matrix final")
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
    candidates = matrix_candidates(args, masks)
    for repeat in range(1, args.repetitions + 1):
        # Rotate order to reduce bias from node load/warm caches between repetitions.
        offset = (repeat - 1) % len(candidates)
        for workers, envs, size, name, mask, backend, device in candidates[offset:] + candidates[:offset]:
            destination = args.output_root.resolve() / args.algorithm / args.matrix / (
                f"w{workers}e{envs}_f{size}_{name}_r{repeat}")
            command = training_command(args, size, destination, mask, backend=backend, collection_device=device,
                                       workers=workers, envs_per_worker=envs)
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            if destination.exists():
                parser.error(f"Output already exists: {destination}; use another --output-root")
            run_training(command, destination, args.profile, args.source_root)
            report_run(destination, args.total_steps, mask, placement["cpu_affinity"],
                       backend=backend, collection_device=device, learner_device=args.learner_device)


if __name__ == "__main__":
    main()
