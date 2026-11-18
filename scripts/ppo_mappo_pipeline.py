#!/usr/bin/env python3
"""Freeze and run the native PPO -> MAPPO pipeline locally or through Slurm."""
import argparse
import csv
from copy import deepcopy
from datetime import datetime
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import statistics

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ray_tune_sources import read_sources, verify_source
from ray_tune_sweeps import digest, identifier, read_spec, ranking, trial_settings
from ray_tune_trials import (atomic_json, current_revision, exclusive,
                             execution_attempt, prepare_settings, stop_group, validate_restoration,
                             verify_native_completion, visible_resources)
from prepare_mappo_pilot_sources import prepare as prepare_sources

LAUNCHER = REPO / "scripts/hpc/ray_tune.sbatch"
TEMPLATES = {stage: REPO / f"scripts/sweeps/ppo_to_mappo_{stage}.yaml" for stage in ("ppo", "mappo")}
STAGES = ("ppo", "mappo", "team", "solo")
SBATCH_OPTIONS = {"--partition", "--account", "--qos", "--nodes", "--cpus-per-task", "--time",
                  "--gres", "--gpus", "--gpus-per-node", "--gpus-per-task", "--mem", "--mem-per-cpu",
                  "--mem-per-gpu", "--constraint", "--reservation", "--nodelist", "--exclude"}


def load_pipeline(root):
    value = json.loads((root / "pipeline.json").read_text())
    if value["digest"] != digest({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError("Frozen pipeline manifest was modified")
    if value["schema_version"] not in {1, 2}:
        raise ValueError("Unsupported pipeline manifest version")
    if value["root"] != str(root) or value["git_sha"] != current_revision():
        raise ValueError("Pipeline path/Git revision differs; use the original clean checkout")
    for stage in value["stages"].values():
        if digest(yaml.safe_load(Path(stage["spec"]).read_text())) != stage["spec_digest"]:
            raise ValueError("Frozen pipeline sweep specification was modified")
    return value


def write_plan(root, pipeline_id, ppo_steps, revision):
    root.mkdir(parents=True, exist_ok=False)
    (root / "specs").mkdir()
    stages = {}
    for stage, template in TEMPLATES.items():
        spec = yaml.safe_load(template.read_text())
        spec["experiment_id"] = f"{pipeline_id}-{stage}"
        spec["wandb"]["group"] = f"{pipeline_id}-{'pretraining' if stage == 'ppo' else 'adaptation'}"
        if stage == "ppo":
            spec["total_steps"] = ppo_steps
        else:
            spec["source_manifest"] = str(root / "sources/manifest.yaml")
        path = root / "specs" / f"{stage}.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        path.chmod(0o444)
        stages[stage] = {"spec": str(path), "spec_digest": digest(spec), "root": str(root / stage),
                         "experiment_id": spec["experiment_id"], "trials": 1 if stage == "ppo" else 6}
    value = {"schema_version": 2, "root": str(root), "pipeline_id": pipeline_id, "git_sha": revision,
             "checkpoint_every_steps": 1024000, "stages": stages,
             "evaluation": {"seed": 30042, "episodes": 100, "team_target_laps": 1,
                            "team_max_steps": 8000, "solo_target_laps": 3, "solo_max_steps": 16000}}
    value["digest"] = digest(value)
    atomic_json(root / "pipeline.json", value)
    (root / "pipeline.json").chmod(0o444)
    return value


def preflight_stage(value, stage):
    """Native CPU/Hydra/source checks in temporary storage; never start Ray."""
    from ray import tune
    spec = read_spec(value["stages"][stage]["spec"], REPO)
    args = argparse.Namespace(experiment_id=spec["experiment_id"], mode=stage, total_steps=spec["total_steps"], seed=42,
        checkpoint_every_steps=value["checkpoint_every_steps"], checkpoint_every_updates=None,
        trial_timeout=0, poll_interval=3, evaluation=True, overrides=[],
        ppo_scenario="scenarios/ppo_lap_completion_pretrain.yaml", mappo_scenario="scenarios/mappo_2v2_completion_scratch.yaml",
        wandb_mode="online", wandb_project=None, wandb_entity=None, wandb_group=None)
    with tempfile.TemporaryDirectory(prefix=f"pipeline-{stage}-preflight-") as temporary:
        checked = prepare_settings(args, spec, Path(temporary), tune, value["git_sha"])
        assert len(checked["trial_plan"]) == value["stages"][stage]["trials"]
    print(f"Native {stage.upper()} preflight passed", file=sys.stderr)


def verify_complete(value, stage):
    """A successful allocation needs actual native budgets, recovery and selection."""
    info = value["stages"][stage]
    root = Path(info["root"])
    settings = json.loads((root / "experiment.json").read_text())
    spec = read_spec(info["spec"], REPO)
    validate_restoration(settings, spec, info["experiment_id"], root)
    results = json.loads((root / "results.json").read_text())
    checked = ranking(results)
    if len(results) != info["trials"] or len(checked["ranked_trials"]) != info["trials"] or checked["diagnostics"]:
        raise ValueError(f"{stage}: missing/failed/incomplete or unrankable trials: {checked['diagnostics']}")
    directories, trial_keys = set(), set()
    for row in checked["ranked_trials"]:
        out = Path(row["training_output_dir"])
        if out.parent != root / "trial_runs" or out in directories:
            raise ValueError("Trial output directories differ or collide")
        directories.add(out)
        identity = json.loads((out / "ray_trial.json").read_text())
        entry = identity["settings"]["sweep_trial"]
        if (entry not in settings["trial_plan"] or identity["settings"] != trial_settings(settings, entry)
                or entry["trial_key"] in trial_keys):
            raise ValueError("Completed trial does not belong to the frozen experiment matrix")
        trial_keys.add(entry["trial_key"])
        summary, final = verify_native_completion(out, identity)
        if row.get("environment_steps", 0) < spec["total_steps"]:
            raise ValueError(f"{stage}: Tune did not report the completed budget: {out}")
        adaptation = final["adaptation"]
        arm = row["adaptation"]
        if adaptation["method"] != arm or (arm == "scratch" and adaptation["source_checkpoint"] is not None):
            raise ValueError("Checkpoint adaptation/source differs from its planned arm")
        if arm in {"lora", "full_finetune"} and adaptation["source_checkpoint"]["sha256"] != identity["settings"]["sweep_trial"]["source"]["sha256"]:
            raise ValueError("Completed checkpoint PPO source differs from the planned immutable source")
        row.update(actual_environment_decisions=summary["environment_steps"],
                   native_last_attempt_training_seconds=summary.get("perf/elapsed_seconds"),
                   latest_checkpoint=str(out / "latest.pt"), final_checkpoint=str(out / "final.pt"),
                   wandb_id=identity["wandb_id"])
    return checked


def completion_sources(value, stage):
    root = Path(value["root"])
    path = root / f"{stage}_complete.json"
    if not path.is_file():
        raise ValueError(f"Complete the {stage} stage before using its artifacts")
    receipt = json.loads(path.read_text())
    sources = list(read_sources(root / "sources/manifest.yaml").values())
    if receipt["pipeline_digest"] != value["digest"] or receipt["sources"] != sources:
        raise ValueError(f"{stage.upper()} completion/source handoff differs from the frozen pipeline")
    for record in sources:
        verify_source(record)
    return sources


def finish_stage(value, stage):
    root = Path(value["root"])
    checked = verify_complete(value, stage)
    if stage == "ppo":
        records = prepare_sources(Path(value["stages"]["ppo"]["root"]), root / "sources",
                                  presets=("mlp_base",), lidar_beams=108,
                                  target_spec=Path(value["stages"]["mappo"]["spec"]))
        preflight_stage(value, "mappo")
        payload = {"pipeline_digest": value["digest"], "sources": records}
    else:
        sources = completion_sources(value, "ppo")
        payload = {"pipeline_digest": value["digest"], "sources": sources,
                   "result_identity": "checkpoint_selection_not_final_test", **checked}
        atomic_json(root / "selection_summary.json", payload)
        for row in checked["ranked_trials"]:
            metrics = row["best_evaluation"]
            print(f"{row['adaptation']} seed={row['seed']} completion={metrics.get('team_both_finished_rate')} "
                  f"collision={metrics.get('team_collision_rate')} decisions={row['actual_environment_decisions']} "
                  f"last_training_attempt_seconds={row['native_last_attempt_training_seconds']} best={row['best_checkpoint']}")
    atomic_json(root / f"{stage}_complete.json", payload)
    record_status(root, stage, "completed")


def sha256(path):
    with Path(path).open("rb") as handle:
        h = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def evaluation_units(value, stage):
    """Derive only from verified native selections; final tests cannot rank models."""
    from agents.common.checkpoints import read_checkpoint
    root = Path(value["root"])
    sources = completion_sources(value, "mappo")
    rows = verify_complete(value, "mappo")["ranked_trials"]
    models = [(f"{r['adaptation']}_seed{r['seed']}", r["best_checkpoint"], r) for r in rows]
    if stage == "solo":
        models.insert(0, ("ppo_seed42", sources[0]["checkpoint"], {"adaptation": "ppo", "seed": 42}))
    units = []
    for name, checkpoint, row in sorted(models):
        payload = read_checkpoint(checkpoint)
        metadata = payload["metadata"]
        if row["adaptation"] not in {"ppo", "scratch"} and metadata["adaptation"]["source_checkpoint"]["sha256"] != sources[0]["sha256"]:
            raise ValueError("Transfer checkpoint no longer matches the immutable PPO source")
        config = deepcopy(metadata["configuration"])
        protocol = value["evaluation"]
        config["evaluation"]["final_test"] = {"seed": protocol["seed"], "episodes": protocol["episodes"],
            "target_laps": protocol[f"{stage}_target_laps"], "max_steps": protocol[f"{stage}_max_steps"]}
        actors = [None] if stage == "team" else (["car_0"] if row["adaptation"] == "ppo" else ["car_0", "car_1"])
        for actor in actors:
            unit = {"name": name + (f"_{actor}" if actor else ""), "stage": stage,
                "adaptation": row["adaptation"], "training_seed": row["seed"],
                "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": payload["_identity"]["sha256"],
                "actor": actor, "config": config, "pipeline_digest": value["digest"]}
            unit["digest"] = digest(unit)
            units.append(unit)
    if len(units) != (6 if stage == "team" else 13):
        raise ValueError("Incorrect final evaluation matrix")
    return units


def checked_report(root, unit):
    folder = root / "evaluations" / unit["stage"] / unit["name"]
    report_path, receipt_path = folder / "evaluation_report.json", folder / "complete.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    report = json.loads(report_path.read_text())
    if (receipt["unit_digest"] != unit["digest"] or receipt["report_sha256"] != sha256(report_path)
            or sha256(unit["checkpoint"]) != unit["checkpoint_sha256"]
            or report["checkpoint_sha256"] != unit["checkpoint_sha256"]):
        raise ValueError("Completed evaluation checkpoint/configuration/report was modified")
    if (report["seeds"] != list(range(30042, 30142)) or len(report["episode_results"]) != 100
            or report["summary"]["episodes"] != 100 or report["protocol"] != "final"
            or report["target_laps"] != (3 if unit["actor"] else 1)
            or report["max_steps"] != (16000 if unit["actor"] else 8000)):
        raise ValueError("Final evaluation protocol/episode count differs")
    if unit["actor"]:
        identity = report["actor_identity"]
        if (identity["source_actor"] != unit["actor"] or identity["critic_loaded"]
                or identity["adapter_active"] != (unit["adaptation"] == "lora")
                or unit["adaptation"] == "lora" and identity["source_adapter_index"] != int(unit["actor"][-1])):
            raise ValueError("Solo actor/adapter identity differs")
    return report


def evaluate_stage(value, stage, device):
    from application.evaluate import run_evaluation
    from loggers.console import ConsoleLogger
    root = Path(value["root"])
    if value["schema_version"] == 1:
        if device != "auto":
            raise ValueError("Historical pipeline manifests require their original CUDA evaluation layout")
        device = "cuda"
    if stage == "solo":
        # Revalidate team receipts before allowing the next dependent stage.
        verify_evaluations(value, "team")
    units = evaluation_units(value, stage)
    target = root / "evaluations" / stage
    target.mkdir(parents=True, exist_ok=True)
    with exclusive(target / ".lock"):
        with execution_attempt(target, "evaluation") as (allocation_path, allocation):
            slurm = bool(os.getenv("SLURM_JOB_ID"))
            resources, affinity = visible_resources(slurm=slurm, batch=slurm)
            if device == "auto":
                device = "cuda" if resources["gpu"] else "cpu"
            allocation.update(nodes=[{"host": socket.gethostname(), **resources, "cpu_affinity": affinity}],
                              ray_node_count=0, allocation_trial_capacity=1, evaluation_device=device)
            atomic_json(allocation_path, allocation)
            if device == "cuda" and not resources["gpu"]:
                raise ValueError("CUDA evaluation requires a visible GPU; select --eval-device cpu")
            manifest = {"pipeline_digest": value["digest"], "units": units}
            path = target / "manifest.json"
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError("Frozen evaluation plan was modified")
            atomic_json(path, manifest)
            for unit in units:
                if checked_report(root, unit) is not None:
                    continue
                folder = target / unit["name"]
                folder.mkdir(parents=True, exist_ok=True)
                atomic_json(folder / "request.json", unit)
                input_path = folder / "evaluation_input.yaml"
                input_path.write_text(yaml.safe_dump(unit["config"], sort_keys=False))
                args = argparse.Namespace(checkpoint=unit["checkpoint"], eval_protocol="final", eval_episodes=None,
                    eval_actor=unit["actor"], eval_device=device, scenario=str(input_path),
                    run_id=f"{value['pipeline_id']}_{stage}_{unit['name']}", output_dir=str(folder))
                run_evaluation(unit["config"], args, ConsoleLogger(verbose=True), REPO / "scenarios")
                atomic_json(folder / "complete.json", {"unit_digest": unit["digest"],
                    "report_sha256": sha256(folder / "evaluation_report.json")})
                checked_report(root, unit)
            reports = verify_evaluations(value, stage)
        if stage == "solo":
            combine_results(value)
        atomic_json(root / f"{stage}_complete.json", {"pipeline_digest": value["digest"], "reports": reports})
        record_status(root, stage, "completed")


def verify_evaluations(value, stage):
    root = Path(value["root"])
    paths = []
    matched = None
    for unit in evaluation_units(value, stage):
        report = checked_report(root, unit)
        if report is None:
            raise ValueError(f"Missing complete independent evaluation: {unit['name']}")
        # Compare actual reset/physics records, not merely a scenario label.
        def normalize_identity(item):
            if isinstance(item, dict):
                return {("solo_vehicle" if key == unit["actor"] else key): normalize_identity(value) for key, value in item.items()}
            if isinstance(item, list):
                return [normalize_identity(value) for value in item]
            return "solo_vehicle" if isinstance(item, str) and item == unit["actor"] else item
        protocol = [{"seed": row["seed"], "map": row["map_bundle"],
                     "spawn": normalize_identity(row["race_record"]["spawn_configuration"]),
                     "physics": normalize_identity(row.get("physics"))} for row in report["episode_results"]]
        if matched is not None and protocol != matched:
            raise ValueError("Final evaluations used different seeds/spawns/physics")
        matched = protocol
        paths.append(str(root / "evaluations" / stage / unit["name"] / "evaluation_report.json"))
    return paths


def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def variation(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.pstdev(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def computational_metrics(out, metadata):
    attempts = []
    for path in sorted((out / "attempts").glob("*/execution.json")):
        attempt = {"path": str(path.parent), **json.loads(path.read_text())}
        placement = path.with_name("attempt.json")
        if placement.exists():
            facts = json.loads(placement.read_text())
            attempt["placement"] = {key: facts.get(key) for key in ("hostname", "slurm_job_id", "slurm_partition",
                "slurm_restart_count", "resources_per_trial", "ray_gpu_ids")}
        attempts.append(attempt)
    wall = sum(a["wall_seconds"] for a in attempts)
    closed = bool(attempts) and all(a["closed"] for a in attempts)
    with (out / "update_metrics.csv").open() as stream:
        updates = list(csv.DictReader(stream))
    perf = {key: variation([numeric(row.get(key)) for row in updates]) for key in (
        "perf/update_seconds", "perf/environment_steps_per_second", "perf/learner_samples_per_second")}
    summary = json.loads((out / "run_summary.json").read_text())
    return {"parameters": metadata["provenance"].get("parameter_counts"),
        "training_process_wall_seconds": wall, "wall_time_is_lower_bound": not closed,
        "overall_completed_decisions_per_second": summary["environment_steps"] / wall if wall else None,
        "overall_learner_samples_per_second": summary.get("agent_steps", 0) / wall if wall else None,
        "wall_time_scope": "sum of trainer process attempts including startup, selection and replay; excludes queue time",
        "attempts": attempts, "update_metrics": perf,
        "last_attempt_native_performance": {k: v for k, v in summary.items() if k.startswith("perf/")},
        "checkpoint_bytes": {name: (out / name).stat().st_size for name in ("best.pt", "latest.pt", "final.pt")}}


def learning_curve(out, attempts):
    records = [json.loads(line) for line in (out / "evaluation_history.jsonl").read_text().splitlines() if line.strip()]
    # A restored completed-update boundary may replay decisions. Keep every raw
    # record on disk; summarize the latest observation at each common clock here.
    latest = {r["environment_steps"]: r for r in records}
    curve = []
    for steps, row in sorted(latest.items()):
        elapsed = None
        attempt = next((a for a in attempts if a["path"] == row.get("training_attempt")), None)
        if attempt and row.get("recorded_unix"):
            elapsed = sum(a["wall_seconds"] for a in attempts if a["started_unix"] < attempt["started_unix"])
            elapsed += row["recorded_unix"] - attempt["started_unix"]
        curve.append({"joint_environment_decisions": steps, "training_wall_seconds": elapsed,
                      "both_finish_rate": row.get("team_both_finished_rate"),
                      "collision_rate": row.get("team_collision_rate", row.get("collision_rate")),
                      "timeout_rate": row.get("timeout_rate")})
    threshold = next((r for r in curve if (r["both_finish_rate"] or 0) >= .9), None)
    return {"selection_curve": curve, "first_observed_90_percent": threshold,
        "at_common_budget": next((r for r in reversed(curve) if r["joint_environment_decisions"] <= 24576000), None),
        "checkpoint_variation": variation([r["both_finish_rate"] for r in curve]),
        "history": str(out / "evaluation_history.jsonl")}


def stage_allocations(root, stage):
    parent = root / stage if stage in TEMPLATES else root / "evaluations" / stage
    records = []
    for attempt in sorted((parent / "attempts").glob("*/")):
        execution, timing, layout = (attempt / name for name in ("execution.json", "allocation_timing.tsv", "allocation.tsv"))
        record = json.loads(execution.read_text()) if execution.is_file() else {}
        record["path"] = str(attempt)
        if layout.is_file():
            record["allocation_tsv"] = str(layout)
        if timing.is_file():
            record["allocation_timing_tsv"] = str(timing)
            with timing.open() as stream:
                timestamps = dict(csv.reader(stream, delimiter="\t"))
            record.update(mode="slurm", **{key: timestamps[key] for key in ("slurm_job_id", "slurm_partition", "slurm_nodelist")
                                          if key in timestamps})
            for key in ("batch_wall_seconds", "allocation_wall_seconds"):
                if key in timestamps:
                    record[key] = int(timestamps[key])
            try:
                record["queue_seconds"] = (datetime.fromisoformat(timestamps["slurm_start_time"])
                                           - datetime.fromisoformat(timestamps["slurm_submit_time"])).total_seconds()
            except (KeyError, ValueError):
                record["queue_seconds"] = None
        records.append(record)
    return records


def combine_results(value):
    from agents.common.checkpoints import read_checkpoint
    root = Path(value["root"])
    verify_evaluations(value, "team")
    verify_evaluations(value, "solo")
    solo_units = evaluation_units(value, "solo")
    baseline_unit = next(u for u in solo_units if u["adaptation"] == "ppo")
    baseline = checked_report(root, baseline_unit)
    rows = []
    team_units = evaluation_units(value, "team")
    for selected in verify_complete(value, "mappo")["ranked_trials"]:
        out = Path(selected["training_output_dir"])
        metadata = read_checkpoint(selected["best_checkpoint"])["metadata"]
        name = f"{selected['adaptation']}_seed{selected['seed']}"
        team_unit = next(u for u in team_units if u["name"] == name)
        team = checked_report(root, team_unit)
        solo, deltas = {}, {}
        keys = ("completion_rate", "strict_clean_completion_rate", "mean_valid_lap_time_s", "mean_net_progress",
                "timeout_rate", "stationary_rate", "zero_progress_rate")
        for aid in ("car_0", "car_1"):
            unit = next(u for u in solo_units if u["name"] == f"{name}_{aid}")
            solo[aid] = {"summary": checked_report(root, unit)["summary"], "report": str(root / "evaluations/solo" / unit["name"] / "evaluation_report.json")}
            deltas[aid] = {key: (solo[aid]["summary"][key] - baseline["summary"][key]
                                if solo[aid]["summary"].get(key) is not None and baseline["summary"].get(key) is not None else None) for key in keys}
        compute = computational_metrics(out, metadata)
        rows.append({"model": "mlp_base", "adaptation": selected["adaptation"], "training_seed": selected["seed"],
            "source": selected.get("source"), "selected_checkpoint": selected["best_checkpoint"],
            "checkpoint_sha256": team["checkpoint_sha256"],
            "training": {"actual_joint_environment_decisions": selected["actual_environment_decisions"],
                         "run_summary": str(out / "run_summary.json"), "wandb_id": selected["wandb_id"]},
            "checkpoint_selection": {"best": selected["best_evaluation"], **learning_curve(out, compute["attempts"])},
            "independent_final_test": {"team": {"summary": team["summary"], "report": str(root / "evaluations/team" / name / "evaluation_report.json")}, "solo": solo},
            "solo_delta_from_ppo": deltas,
            "solo_comparison_kind": "scratch_comparison_baseline" if selected["adaptation"] == "scratch" else "pretrained_retention",
            "actor_difference_car1_minus_car0": {key: solo["car_1"]["summary"][key] - solo["car_0"]["summary"][key]
                if solo["car_1"]["summary"].get(key) is not None and solo["car_0"]["summary"].get(key) is not None else None for key in keys},
            "computation": compute})
    ppo_row = verify_complete(value, "ppo")["ranked_trials"][0]
    ppo_metadata = read_checkpoint(baseline_unit["checkpoint"])["metadata"]
    atomic_json(root / "summary.json", {"pipeline_digest": value["digest"], "ppo_source": baseline_unit["checkpoint"],
        "stage_allocations": {stage: stage_allocations(root, stage) for stage in STAGES},
        "ppo_training": {"actual_environment_decisions": ppo_row["actual_environment_decisions"],
                         "computation": computational_metrics(Path(ppo_row["training_output_dir"]), ppo_metadata)},
        "ppo_solo_baseline": {"checkpoint_sha256": baseline["checkpoint_sha256"], "summary": baseline["summary"],
                              "report": str(root / "evaluations/solo/ppo_seed42_car_0/evaluation_report.json")},
        "trials": rows, "seed_variation_final_both_finish": {
            arm: variation([r["independent_final_test"]["team"]["summary"]["team_both_finished_rate"] for r in rows if r["adaptation"] == arm])
            for arm in ("scratch", "full_finetune", "lora")},
        "seed_variation_selection_common_budget": {
            arm: variation([r["checkpoint_selection"]["at_common_budget"]["both_finish_rate"]
                            for r in rows if r["adaptation"] == arm and r["checkpoint_selection"]["at_common_budget"]])
            for arm in ("scratch", "full_finetune", "lora")},
        "interpretation": "Selection and independent final tests are separate; stationary episodes alone do not establish catastrophic forgetting."})


def record_status(root, stage, status):
    atomic_json(root / f"{stage}_status.json", {"stage": stage, "status": status, "updated_unix": time.time(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "restart_count": os.getenv("SLURM_RESTART_COUNT", "0")})


def stage_complete(value, stage):
    path = Path(value["root"]) / f"{stage}_complete.json"
    if not path.is_file():
        return False
    if stage in TEMPLATES:
        completion_sources(value, stage)
        verify_complete(value, stage)
    else:
        receipt = json.loads(path.read_text())
        if receipt != {"pipeline_digest": value["digest"], "reports": verify_evaluations(value, stage)}:
            raise ValueError(f"{stage} evaluation completion manifest differs")
    return True


def require_previous_stage(value, stage):
    index = STAGES.index(stage)
    if index and not stage_complete(value, STAGES[index - 1]):
        raise ValueError(f"Complete the {STAGES[index - 1]} stage before running {stage}")


def stage_details(value, stage):
    require_previous_stage(value, stage)
    complete = stage_complete(value, stage)
    env = {"PIPELINE_STAGE_COMPLETE": int(complete)}
    if stage not in TEMPLATES:
        return complete, env
    info = value["stages"][stage]
    env.update(SWEEP_SPEC=info["spec"], EXPERIMENT_ID=info["experiment_id"], RAY_TUNE_JOB_ROOT=info["root"],
               RESTORE=int((Path(info["root"]) / "experiment.json").exists()),
               RAY_TUNE_CHECKPOINT_STEPS=value["checkpoint_every_steps"], RAY_TUNE_TRIAL_TIMEOUT=0)
    return complete, env


def stage_environment(value, stage):
    complete, env = stage_details(value, stage)
    legacy = value["schema_version"] == 1
    restarts = value["max_restarts"] if legacy else int(os.getenv("PIPELINE_MAX_RESTARTS", "20"))
    if restarts < 0:
        raise ValueError("PIPELINE_MAX_RESTARTS must be nonnegative")
    env["PIPELINE_MAX_RESTARTS"] = restarts
    if stage in TEMPLATES:
        info = value["stages"][stage]
        for key in ("nodes", "cpus_per_node", "gpus_per_node"):
            env[f"RAY_TUNE_FIXED_{key.upper()}"] = info[key] if legacy else ""
        env["PIPELINE_LEGACY_LAYOUT"] = int(legacy)
    record_status(Path(value["root"]), stage, "completed" if complete else "running")
    for key, item in env.items():
        print(f"export {key}={shlex.quote(str(item))}")


def run_local(value, stages, device):
    if value["schema_version"] == 1:
        raise ValueError("Historical pipeline manifests require their original fixed Slurm layout")
    root = Path(value["root"])
    for stage in stages:
        with exclusive(root / f".{stage}_allocation.lock") as lease:
            complete, settings = stage_details(value, stage)
            if complete:
                print(f"{stage} already verified")
                continue
            record_status(root, stage, "running")
            try:
                if stage in TEMPLATES:
                    env = {**os.environ, **{key: str(value) for key, value in settings.items()}}
                    with subprocess.Popen([sys.executable, str(REPO / "scripts/ray_tune_trials.py")],
                                          cwd=REPO, env=env, start_new_session=True, pass_fds=(lease.fileno(),)) as process:
                        try:
                            if process.wait():
                                raise RuntimeError(f"{stage} Ray Tune driver exited {process.returncode}")
                        except (KeyboardInterrupt, SystemExit):
                            if process.poll() is None:
                                process.send_signal(signal.SIGINT)
                                try:
                                    process.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    pass
                            raise
                        finally:
                            if process.poll() is None:
                                stop_group(process)
                    finish_stage(load_pipeline(root), stage)
                else:
                    evaluate_stage(value, stage, device)
            except (KeyboardInterrupt, SystemExit):
                record_status(root, stage, "interrupted")
                raise
            except Exception:
                record_status(root, stage, "failed")
                raise


def submission_command(value, stage, dependency=None, *, options=(), walltime=None, max_restarts=None, device="auto"):
    if value["schema_version"] == 1:
        if options or walltime is not None or max_restarts is not None:
            raise ValueError("Historical pipeline manifests require their original fixed Slurm layout")
        info = value["stages"].get(stage, {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 4})
        defaults = {"--partition": "gpu", "--account": "q1wang", "--qos": "normal", "--nodes": info["nodes"],
                    "--cpus-per-task": info["cpus_per_node"], "--gres": f"gpu:{info['gpus_per_node']}",
                    "--time": value["walltime"]}
        max_restarts = value["max_restarts"]
    else:
        resources = (read_spec(value["stages"][stage]["spec"], REPO)["resources_per_trial"]
                     if stage in TEMPLATES else {"cpu": 4, "gpu": 0})
        defaults = {"--nodes": 1, "--cpus-per-task": resources["cpu"], "--time": walltime or "24:00:00"}
        if resources["gpu"] and not any(key in options for key in ("--gres", "--gpus", "--gpus-per-node", "--gpus-per-task")):
            defaults["--gres"] = f"gpu:{resources['gpu']}"
        defaults.update(options)
    exports = (f"ALL,PIPELINE_ROOT={value['root']},PIPELINE_STAGE={stage},"
               f"PIPELINE_MAX_RESTARTS={20 if max_restarts is None else max_restarts},PIPELINE_EVAL_DEVICE={device}")
    command = ["sbatch", "--parsable", "--hold", "--ntasks-per-node=1",
        *(f"{key}={item}" for key, item in defaults.items()),
        f"--job-name={value['pipeline_id']}-{stage}", "--signal=B:USR1@240", "--requeue", "--open-mode=append",
        f"--output={value['root']}/{stage}-%j.log", f"--export={exports}"]
    if dependency:
        command.extend([f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"])
    return [*command, str(LAUNCHER)]


def queue_pipeline(value, *, resume=False, stage=None, options=None, walltime=None, max_restarts=None, device="auto"):
    root = Path(value["root"])
    options = options or {}
    stages = [stage] if stage else list(STAGES)
    with exclusive(root / ".submission.lock"):
        if stage:
            require_previous_stage(value, stage)
        while stages and stage_complete(value, stages[0]):
            stages.pop(0)
        if not stages:
            if stage in (None, "solo"):
                combine_results(value)
            return
        commands = {item: submission_command(value, item, options=options.get(item, {}),
                                              walltime=walltime, max_restarts=max_restarts, device=device) for item in stages}
        if (root / "jobs.json").exists():
            previous = json.loads((root / "jobs.json").read_text())
            if not resume and (not stage or stage in previous["jobs"]):
                raise ValueError("Pipeline already submitted; use --resume after a terminal failure")
            if previous["pipeline_digest"] != value["digest"]:
                raise ValueError("Slurm job record belongs to a different pipeline")
            ids = set(previous["jobs"].values())
            live = subprocess.check_output(["squeue", "--all", "--me", "--noheader", "--format=%i %T %r"],
                                           text=True) if ids else ""
            states = [row for row in (line.split(maxsplit=2) for line in live.splitlines()) if row and row[0] in ids]
            for _, state, reason in states:
                if state != "PENDING" or reason not in {"DependencyNeverSatisfied", "Dependency"}:
                    raise ValueError("Pipeline jobs are still active; automatic requeue owns recovery")
            if states and not any(reason == "DependencyNeverSatisfied" for _, _, reason in states):
                raise ValueError("Pipeline has live dependencies; refusing to replace active work")
            # A dependency that has already failed cannot be repaired by merely
            # requeuing its parent. Replace only the old pending dependent job.
            for job, _, _ in states:
                subprocess.run(["scancel", job], check=True)
        # Slurm's ALL export gives existing values priority over explicit ones.
        env = os.environ.copy()
        for key in ("PIPELINE_ROOT", "PIPELINE_STAGE", "PIPELINE_MAX_RESTARTS", "PIPELINE_EVAL_DEVICE"):
            env.pop(key, None)
        jobs = {}
        try:
            dependency = None
            for stage, command in commands.items():
                if dependency:
                    command[-1:-1] = [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
                output = subprocess.check_output(command, text=True, env=env).strip()
                job = output.split(";", 1)[0]
                if not job.isdigit():
                    raise ValueError(f"Unexpected sbatch job identity: {output!r}")
                jobs[stage] = job
                dependency = job
                atomic_json(root / "jobs.json", {"jobs": jobs, "status": "held", "pipeline_digest": value["digest"]})
            for job in jobs.values():
                subprocess.run(["scontrol", "release", job], check=True)
            atomic_json(root / "jobs.json", {"jobs": jobs, "status": "released", "pipeline_digest": value["digest"]})
        except Exception:
            for job in jobs.values():
                subprocess.run(["scancel", job], check=False)
            atomic_json(root / "jobs.json", {"jobs": jobs, "status": "submission_failed", "pipeline_digest": value["digest"]})
            raise
    print(json.dumps({"pipeline_root": str(root), "slurm_jobs": jobs}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--create", action="store_true", help="Freeze and preflight a new pipeline without running it")
    mode.add_argument("--run", action="store_true", help="Run a local stage or all remaining stages")
    mode.add_argument("--submit", action="store_true", help="Submit one stage or a dependent Slurm chain")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, help="Existing pipeline root after terminal failure; retains frozen settings")
    parser.add_argument("--pipeline-id")
    parser.add_argument("--ppo-steps", type=int)
    parser.add_argument("--walltime", help="Default Slurm walltime for this submission")
    parser.add_argument("--max-restarts", type=int, help="Maximum automatic Slurm requeues per job")
    parser.add_argument("--sbatch-option", action="append", default=[], metavar="STAGE=--OPTION=VALUE",
                        help="Set a standard Slurm resource option for one stage (repeatable)")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "cuda"), default=os.getenv("PIPELINE_EVAL_DEVICE", "auto"))
    parser.add_argument("--stage-action", choices=("environment", "finish", "evaluate", "interrupted", "failed"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    for name in ("root", "resume"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    os.chdir(REPO)
    if args.eval_device not in {"auto", "cpu", "cuda"}:
        parser.error("PIPELINE_EVAL_DEVICE must be auto, cpu, or cuda")
    if args.stage_action:
        if args.root is None or args.stage is None:
            parser.error("Stage actions require --root and --stage")
        value = load_pipeline(args.root)
        if args.stage_action == "environment":
            stage_environment(value, args.stage)
        elif args.stage_action == "finish" and args.stage in TEMPLATES:
            finish_stage(value, args.stage)
        elif args.stage_action == "evaluate" and args.stage in {"team", "solo"}:
            evaluate_stage(value, args.stage, args.eval_device)
        elif args.stage_action in {"failed", "interrupted"}:
            record_status(args.root, args.stage, args.stage_action)
        else:
            parser.error("Invalid action for this stage")
        return
    if args.resume and (not args.submit or args.root is not None or args.pipeline_id is not None):
        parser.error("--resume ROOT requires --submit and selects the original pipeline")
    if (args.ppo_steps is not None and args.ppo_steps < 1
            or args.max_restarts is not None and args.max_restarts < 0):
        parser.error("Positive PPO budget and nonnegative max-restarts are required")
    if args.walltime is not None and (not re.fullmatch(r"\d{2,3}:[0-5]\d:[0-5]\d", args.walltime)
                                     or args.walltime == "00:00:00"):
        parser.error("Use a positive --walltime in HH:MM:SS format")
    options = {}
    for raw in args.sbatch_option:
        stage, _, option = raw.partition("=")
        key, _, item = option.partition("=")
        if stage not in STAGES or key not in SBATCH_OPTIONS or not item:
            parser.error(f"Expected STAGE=--OPTION=VALUE for a Slurm resource option: {raw!r}")
        options.setdefault(stage, {})[key] = item
    if (args.create or args.run) and (options or args.walltime is not None or args.max_restarts is not None):
        parser.error("Slurm options apply to submissions and previews")
    pipeline_id = identifier(args.pipeline_id or "ppo-mappo-mlp108-v1")
    root = (args.resume or args.root or REPO / "outputs/ppo_mappo_pipeline" / pipeline_id).resolve()
    stages = [args.stage] if args.stage else STAGES
    if not (args.create or args.run) and any(character in str(root) for character in ",\n"):
        parser.error("Pipeline paths cannot contain commas or newlines (Slurm export syntax)")
    ppo_steps = args.ppo_steps or 120000000
    if args.create or (args.submit and not (root / "pipeline.json").exists() and not args.resume):
        if args.stage:
            parser.error("Create the pipeline before running or submitting an individual stage")
        value = write_plan(root, pipeline_id, ppo_steps, current_revision())
        preflight_stage(value, "ppo")
        if args.create:
            print(root)
            return
    elif args.run or args.submit or (root / "pipeline.json").exists():
        if args.ppo_steps is not None:
            parser.error("An existing pipeline retains its frozen PPO budget")
        value = load_pipeline(root)
    else:
        with tempfile.TemporaryDirectory(prefix="ppo-mappo-preview-") as temporary:
            value = write_plan(Path(temporary) / "plan", pipeline_id, ppo_steps, "preview")
            preflight_stage(value, "ppo")
            value["root"] = str(root)
            preview(value, stages, options, args.walltime, args.max_restarts, args.eval_device)
        return
    if args.run:
        def terminate(signum, _):
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            run_local(value, stages, args.eval_device)
        finally:
            signal.signal(signal.SIGTERM, previous)
    elif args.submit:
        queue_pipeline(value, resume=bool(args.resume), stage=args.stage, options=options,
                       walltime=args.walltime, max_restarts=args.max_restarts, device=args.eval_device)
    else:
        preview(value, stages, options, args.walltime, args.max_restarts, args.eval_device)


def preview(value, stages, options, walltime, max_restarts, device):
    dependency = None
    for stage in stages:
        print(shlex.join(submission_command(value, stage, dependency, options=options.get(stage, {}),
                                            walltime=walltime, max_restarts=max_restarts, device=device)))
        dependency = f"{stage.upper()}_JOB_ID"
    print("Preview only. PPO -> source verification -> MAPPO -> team finals -> solo finals -> summary.json.")


if __name__ == "__main__":
    main()
