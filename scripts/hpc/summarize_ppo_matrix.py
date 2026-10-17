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


def slurm_accounting(job):
    values = dict(slurm_state=None, slurm_exit_code=None, slurm_reason=None,
                  allocation_seconds=None, accounting_error=None)
    if not job.isdigit():
        return values
    try:
        cp = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job,
             "--format=JobIDRaw,ElapsedRaw,State%40,ExitCode,Reason%100"],
            capture_output=True, text=True, check=True, timeout=15)
        for line in cp.stdout.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 5 and parts[0] == job:
                values.update(slurm_state=parts[2], slurm_exit_code=parts[3],
                              slurm_reason=parts[4] if parts[4] not in {"", "None", "Unknown", "(null)"} else None,
                              allocation_seconds=int(parts[1]))
                return values
        values["accounting_error"] = "allocation not found in sacct"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        values["accounting_error"] = str(exc)
    return values


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
        values.update(slurm_accounting(record["job_id"]))
        values["missing_results"] = not (root / "results.json").is_file()
        values["failure_reason"] = None
        values["aggregate_steps_per_second"] = None
        values["allocated_cpu_hours"] = None
        values["allocated_gpu_hours"] = None
        results = []
        if not values["missing_results"]:
            try:
                results = json.loads((root / "results.json").read_text())
                if not isinstance(results, list) or any(not isinstance(r, dict) for r in results):
                    raise ValueError("results.json must contain a list of trial results")
            except (OSError, ValueError) as exc:
                values["failure_reason"] = f"invalid results.json: {exc}"
                results = []
        if results:
            completed = [r for r in results if r.get("status") == "completed" and not r.get("error")]
            values["completed_trials"] = len(completed)
            values["all_trials_completed"] = len(completed) == int(record["trials"])
            errors = [str(r["error"]) for r in results if r.get("error")]
            if errors:
                values["failure_reason"] = "; ".join(errors)
            native, steady = [], []
            # Keep failed/partial profiles out of throughput and scaling tables.
            successful = (values["all_trials_completed"] and
                          values["slurm_state"] in {None, "COMPLETED"} and
                          values["slurm_exit_code"] in {None, "0:0"} and not values["failure_reason"])
            for result in completed if successful else []:
                summary_path = Path(result["training_output_dir"]) / "run_summary.json"
                if not summary_path.is_file():
                    values["failure_reason"] = f"missing native summary: {summary_path}"
                    continue
                try:
                    summary = json.loads(summary_path.read_text())
                except (OSError, ValueError) as exc:
                    values["failure_reason"] = f"invalid native summary: {exc}"
                    continue
                if "perf/environment_steps_per_second" in summary:
                    native.append(float(summary["perf/environment_steps_per_second"]))
                if "perf/steady_environment_steps_per_second" in summary:
                    steady.append(float(summary["perf/steady_environment_steps_per_second"]))
            successful = successful and not values["failure_reason"]
            if native and successful:
                values["mean_native_steps_per_second"] = round(statistics.mean(native), 2)
            if steady and successful:
                values["mean_steady_steps_per_second"] = round(statistics.mean(steady), 2)
            if values["allocation_seconds"] and successful:
                wall = values["allocation_seconds"]
                total = sum(int(r["environment_steps"]) for r in completed)
                values["aggregate_steps_per_second"] = round(total / wall, 2)
                values["allocated_cpu_hours"] = round(values["cpus_total"] * wall / 3600, 3)
                values["allocated_gpu_hours"] = round(values["gpus_total"] * wall / 3600, 3)
        elif values["missing_results"]:
            # A driver interrupted before results.json may have completed trials.
            for path in (root / "trial_runs").glob("*/run_summary.json"):
                try:
                    summary = json.loads(path.read_text())
                    if (path.parent / "final.pt").is_file() and summary.get("environment_steps", 0) >= int(record["steps_per_trial"]):
                        values["completed_trials"] += 1
                except (OSError, ValueError):
                    pass
        if values["slurm_state"] not in {None, "COMPLETED"}:
            detail = f"{values['slurm_state']} exit={values['slurm_exit_code']}"
            if values["slurm_reason"]:
                detail += ": " + values["slurm_reason"]
            values["failure_reason"] = "; ".join(filter(None, (detail, values["failure_reason"])))
        if values["missing_results"]:
            values["failure_reason"] = "; ".join(filter(None, (values["failure_reason"], "missing results.json")))
        elif not values["all_trials_completed"] and not values["failure_reason"]:
            values["failure_reason"] = "not all trials completed"
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
              f"state={row['slurm_state']} exit={row['slurm_exit_code']} "
              f"wall={row['allocation_seconds']}s missing_results={row['missing_results']} "
              f"reason={row['failure_reason'] or row['accounting_error']} aggregate={row['aggregate_steps_per_second']} "
              f"steady/trial={row['mean_steady_steps_per_second']} speedup={row['speedup_vs_w16_n1g1']}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 scripts/hpc/summarize_ppo_matrix.py outputs/hpc_profile_matrix/TAG/jobs.tsv")
    run(Path(sys.argv[1]).expanduser().resolve())
