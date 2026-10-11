"""Collect matched CPU/CUDA throughput runs into JSON and CSV reports."""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path


def read_run(directory, device):
    directory = Path(directory)
    snapshot = json.loads((directory / "config_snapshot.json").read_text())
    summary = json.loads((directory / "run_summary.json").read_text())
    execution = snapshot["provenance"]["execution"]
    for key in ("policy_device", "update_device"):
        if execution[key].split(":")[0] != device:
            raise ValueError(f"{directory}: expected {key}={device}, got {execution[key]}")
    for key in ("measure_stages", "measure_mpc", "measure_hybrid"):
        if execution[key]:
            raise ValueError(f"{directory}: {key} must be disabled")
    if any(key.startswith("perf/stages/") for key in summary):
        raise ValueError(f"{directory}: summary contains detailed stage profiling")
    config = deepcopy(snapshot["config"])
    target = config["experiment"]["total_steps"]
    if summary["environment_steps"] != target:
        raise ValueError(f"{directory}: incomplete run; expected {target} decisions")
    # CLI provenance differs with device and destination; compare effective values.
    config.pop("_configuration", None)
    config["execution"]["params"].pop("device")
    config["training_defaults"].pop("device")
    with (directory / "update_metrics.csv").open(newline="") as handle:
        updates = list(csv.DictReader(handle))
    row = {
        "device": device,
        "directory": str(directory),
        "environment_steps": summary["environment_steps"],
        "physics_steps": summary["physics_steps"],
        "learner_samples": summary["agent_steps"],
        "completed_episodes": summary["completed_episodes"],
        "updates": summary["updates"],
        "optimizer_steps": summary["optimizer_steps"],
        "elapsed_seconds": summary["perf/elapsed_seconds"],
        "environment_steps_per_second": summary["perf/environment_steps_per_second"],
        "physics_steps_per_second": summary["perf/physics_steps_per_second"],
        "learner_samples_per_second": summary["perf/learner_samples_per_second"],
        "optimizer_steps_per_second": summary["perf/optimizer_steps_per_second"],
        "collection_seconds": sum(float(r["perf/collection_seconds"]) for r in updates),
        "update_seconds": sum(float(r["perf/update_seconds"]) for r in updates),
        "logging_seconds": summary.get("perf/logging_seconds", 0),
        "evaluation_seconds": summary.get("perf/evaluation_seconds", 0),
        "checkpoint_seconds": summary.get("perf/checkpoint_seconds", 0),
    }
    metadata = {"runtime": snapshot["provenance"]["runtime"], "execution": execution}
    return config, row, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", required=True,
                        metavar=("NAME", "CPU_DIR", "CUDA_DIR"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Output prefix; writes PREFIX.json and PREFIX.csv")
    args = parser.parse_args()
    rows, comparisons = [], []
    names = set()
    for name, cpu_dir, cuda_dir in args.pair:
        if name in names:
            parser.error(f"Duplicate pair name: {name}")
        names.add(name)
        cpu_config, cpu, cpu_metadata = read_run(cpu_dir, "cpu")
        cuda_config, cuda, cuda_metadata = read_run(cuda_dir, "cuda")
        if cpu_config != cuda_config:
            raise ValueError(f"{name}: effective configurations differ beyond device selection")
        for row in (cpu, cuda):
            rows.append({"pair": name, **row})
        ratio = cuda["environment_steps_per_second"] / cpu["environment_steps_per_second"]
        comparisons.append({
            "name": name,
            "configuration_match_verified": True,
            "configuration_without_device_or_cli_provenance": cpu_config,
            "runs": {"cpu": cpu, "cuda": cuda},
            "runtime_and_execution": {"cpu": cpu_metadata, "cuda": cuda_metadata},
            "cuda_to_cpu_throughput_ratio": ratio,
            "cuda_throughput_percent_change": 100 * (ratio - 1),
        })
    report = {
        "comparisons": comparisons,
        "definitions": {
            "environment_steps": "Joint decisions, independent of active learner count.",
            "learner_samples": "Active learner decisions; can differ between device runs.",
            "elapsed_seconds": "Trainer wall time, including initialization and in-run hooks; excludes application setup and final artifact saving.",
            "collection_seconds": "Sum of per-update collection wall times excluding service time; first interval includes collector initialization.",
            "update_seconds": "Sum of per-update wall times, with CUDA synchronization at update boundaries.",
        },
        "limitations": [
            "One run per device does not estimate run-to-run variability.",
            "Identical configurations and seeds can produce different CPU/CUDA trajectories and active learner workloads.",
            "PPO and MAPPO have different learning workloads; compare devices within each pair.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(args.output) + ".json")
    csv_path = Path(str(args.output) + ".csv")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Pair | Device | Decisions/s | Learner samples/s | Elapsed s | Collection s | Update s")
    for row in rows:
        print(f"{row['pair']} | {row['device']} | {row['environment_steps_per_second']:.1f} | "
              f"{row['learner_samples_per_second']:.1f} | {row['elapsed_seconds']:.2f} | "
              f"{row['collection_seconds']:.2f} | {row['update_seconds']:.2f}")
    print(f"Saved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
