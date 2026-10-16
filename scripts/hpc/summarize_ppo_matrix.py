#!/usr/bin/env python3
"""Summarize fixed-workload PPO Ray profiling from the submitter's jobs.tsv.

Runs under system Python (no torch/Ray required). Slurm accounting is used for
allocation makespan; native summaries supply independent per-trial throughput.
"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def elapsed_seconds(job):
    if not job.isdigit():
        return None
    try:
        cp = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,ElapsedRaw,State"],
            capture_output=True, text=True, check=True, timeout=15)
        for line in cp.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 3 and parts[0] == job and parts[2].startswith("COMPLETED"):
                return int(parts[1])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def run(manifest):
    repo = Path(__file__).resolve().parents[2]
    rows = []
    with manifest.open(newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    for record in records:
        root = repo / "outputs" / ("ray_tune_" + record["experiment_id"])
        values = dict(record)
        values["gpus_total"] = int(record["nodes"]) * int(record["gpus_per_node"])
        values["cpus_total"] = int(record["nodes"]) * int(record["cpus_per_node"])
        values["env_slots_per_trial"] = int(record["workers"]) * int(record["envs_per_worker"])
        values["completed_trials"] = 0
        values["all_trials_completed"] = False
        values["mean_native_steps_per_second"] = None
        values["mean_steady_steps_per_second"] = None
        values["allocation_seconds"] = elapsed_seconds(record["job_id"])
        values["aggregate_steps_per_second"] = None
        values["allocated_cpu_hours"] = None
        values["allocated_gpu_hours"] = None
        if (root / "results.json").exists():
            results = json.loads((root / "results.json").read_text())
            completed = [r for r in results if r.get("status") == "completed" and not r.get("error")]
            values["completed_trials"] = len(completed)
            values["all_trials_completed"] = len(completed) == int(record["trials"])
            native, steady = [], []
            for result in completed:
                summary_path = Path(result["training_output_dir"]) / "run_summary.json"
                if not summary_path.is_file():
                    continue
                summary = json.loads(summary_path.read_text())
                if "perf/environment_steps_per_second" in summary:
                    native.append(float(summary["perf/environment_steps_per_second"]))
                if "perf/steady_environment_steps_per_second" in summary:
                    steady.append(float(summary["perf/steady_environment_steps_per_second"]))
            if native:
                values["mean_native_steps_per_second"] = round(statistics.mean(native), 2)
            if steady:
                values["mean_steady_steps_per_second"] = round(statistics.mean(steady), 2)
            if values["allocation_seconds"] and values["all_trials_completed"]:
                wall = values["allocation_seconds"]
                total = sum(int(r["environment_steps"]) for r in completed)
                values["aggregate_steps_per_second"] = round(total / wall, 2)
                values["allocated_cpu_hours"] = round(values["cpus_total"] * wall / 3600, 3)
                values["allocated_gpu_hours"] = round(values["gpus_total"] * wall / 3600, 3)
        rows.append(values)
    baseline = next((r["aggregate_steps_per_second"] for r in rows
                     if r["profile"] == "w16_n1g1" and r["aggregate_steps_per_second"]), None)
    for row in rows:
        row["speedup_vs_w16_n1g1"] = (round(row["aggregate_steps_per_second"] / baseline, 3)
                                        if baseline and row["aggregate_steps_per_second"] else None)
    fields = list(rows[0]) if rows else []
    destination = manifest.with_name("summary.csv")
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['profile']:<11} trials={row['completed_trials']}/{row['trials']} "
              f"GPU={row['gpus_total']} CPU={row['cpus_total']} "
              f"wall={row['allocation_seconds']}s aggregate={row['aggregate_steps_per_second']} "
              f"steady/trial={row['mean_steady_steps_per_second']} speedup={row['speedup_vs_w16_n1g1']}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 scripts/hpc/summarize_ppo_matrix.py outputs/hpc_profile_matrix/TAG/jobs.tsv")
    run(Path(sys.argv[1]).expanduser().resolve())
