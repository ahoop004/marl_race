#!/usr/bin/env python3
"""Submit/finalize the existing native Ray sweeps as a dependent PPO -> MAPPO run.

Default is a CPU-only dry run; --submit releases held PPO, MAPPO, team-final and
solo-final jobs after their afterok dependencies are recorded. Native Ray owns
training; the shared evaluator owns all final races.
"""
import argparse
import csv
from copy import deepcopy
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import statistics

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ray_tune_sources import configuration_digest, read_sources, verify_source
from ray_tune_sweeps import digest, identifier, read_spec, ranking
from ray_tune_trials import (atomic_json, checkpoint_payload, current_revision, exclusive,
                             prepare_settings, validate_restoration)
from prepare_mappo_pilot_sources import prepare as prepare_sources

LAUNCHER = REPO / "scripts/hpc/ray_tune_matrix.sbatch"
TEMPLATES = {stage: REPO / f"scripts/sweeps/ppo_to_mappo_{stage}.yaml" for stage in ("ppo", "mappo")}
STAGES = ("ppo", "mappo", "team", "solo")


def load_pipeline(root):
    value = json.loads((root / "pipeline.json").read_text())
    if value["digest"] != digest({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError("Frozen pipeline manifest was modified")
    if value["root"] != str(root) or value["git_sha"] != current_revision():
        raise ValueError("Pipeline path/Git revision differs; use the original clean checkout")
    for stage in value["stages"].values():
        if digest(yaml.safe_load(Path(stage["spec"]).read_text())) != stage["spec_digest"]:
            raise ValueError("Frozen pipeline sweep specification was modified")
    return value


def write_plan(root, pipeline_id, ppo_steps, walltime, max_restarts, revision):
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
                         "experiment_id": spec["experiment_id"], "trials": 1 if stage == "ppo" else 6,
                         "nodes": 1 if stage == "ppo" else 2, "gpus_per_node": 1 if stage == "ppo" else 2,
                         "cpus_per_node": 24}
    value = {"schema_version": 1, "root": str(root), "pipeline_id": pipeline_id, "git_sha": revision,
             "checkpoint_every_steps": 1024000, "walltime": walltime, "max_restarts": max_restarts,
             "stages": stages,
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
    from agents.common.checkpoints import read_checkpoint
    info = value["stages"][stage]
    root = Path(info["root"])
    settings = json.loads((root / "experiment.json").read_text())
    spec = read_spec(info["spec"], REPO)
    validate_restoration(settings, spec, info["experiment_id"], root)
    results = json.loads((root / "results.json").read_text())
    checked = ranking(results)
    if len(results) != info["trials"] or len(checked["ranked_trials"]) != info["trials"] or checked["diagnostics"]:
        raise ValueError(f"{stage}: missing/failed/incomplete or unrankable trials: {checked['diagnostics']}")
    directories = set()
    for row in checked["ranked_trials"]:
        out = Path(row["training_output_dir"])
        if out.parent != root / "trial_runs" or out in directories:
            raise ValueError("Trial output directories differ or collide")
        directories.add(out)
        identity = json.loads((out / "ray_trial.json").read_text())
        if identity["settings"]["sweep_trial"] not in settings["trial_plan"]:
            raise ValueError("Completed trial does not belong to the frozen experiment matrix")
        summary = json.loads((out / "run_summary.json").read_text())
        latest = checkpoint_payload(out / "latest.pt", identity)
        final = read_checkpoint(out / "final.pt")
        budget = spec["total_steps"]
        if (row.get("environment_steps", 0) < budget or summary["environment_steps"] < budget
                or latest["training"]["progress"]["environment_steps"] < budget
                or final["metadata"]["progress"]["environment_steps"] < budget):
            raise ValueError(f"{stage}: actual native budget not reached: {out}")
        if (final["metadata"]["provenance"].get("run_id") != identity["run_id"]
                or configuration_digest(final["metadata"]["configuration"]) != identity["settings"]["sweep_trial"]["config_digest"]
                or final["metadata"]["adaptation"] != latest["metadata"]["adaptation"]):
            raise ValueError(f"{stage}: final checkpoint identity/configuration/source differs")
        adaptation = final["metadata"]["adaptation"]
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


def finish_stage(root, stage):
    value = load_pipeline(root)
    checked = verify_complete(value, stage)
    if stage == "ppo":
        records = prepare_sources(Path(value["stages"]["ppo"]["root"]), root / "sources",
                                  presets=("mlp_base",), lidar_beams=108,
                                  target_spec=Path(value["stages"]["mappo"]["spec"]))
        preflight_stage(value, "mappo")
        payload = {"pipeline_digest": value["digest"], "sources": records}
    else:
        sources = list(read_sources(root / "sources/manifest.yaml").values())
        for record in sources:
            verify_source(record)
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
    completed = json.loads((root / "mappo_complete.json").read_text())
    if completed["pipeline_digest"] != value["digest"]:
        raise ValueError("MAPPO completion manifest differs")
    rows = verify_complete(value, "mappo")["ranked_trials"]
    sources = list(read_sources(root / "sources/manifest.yaml").values())
    for record in sources:
        verify_source(record)
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


def evaluate_stage(root, stage):
    from application.evaluate import run_evaluation
    from loggers.console import ConsoleLogger
    value = load_pipeline(root)
    if stage == "solo":
        # Revalidate team receipts before allowing the next dependent stage.
        verify_evaluations(value, "team")
    units = evaluation_units(value, stage)
    target = root / "evaluations" / stage
    target.mkdir(parents=True, exist_ok=True)
    with exclusive(target / ".lock"):
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
                eval_actor=unit["actor"], scenario=str(input_path),
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
        attempts.append({"path": str(path.parent), **json.loads(path.read_text())})
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


def stage_environment(root, stage):
    value = load_pipeline(root)
    if stage in {"team", "solo"}:
        if stage == "team":
            verify_complete(value, "mappo")
        else:
            verify_evaluations(value, "team")
        record_status(root, stage, "running")
        print(f"export PIPELINE_MAX_RESTARTS={value['max_restarts']}")
        return
    info = value["stages"][stage]
    if stage == "mappo":
        completed = json.loads((root / "ppo_complete.json").read_text())
        if completed["pipeline_digest"] != value["digest"] or list(read_sources(root / "sources/manifest.yaml").values()) != completed["sources"]:
            raise ValueError("PPO completion/source handoff differs from the frozen pipeline")
        for record in completed["sources"]:
            verify_source(record)
    complete = (root / f"{stage}_complete.json").exists()
    if complete:
        verify_complete(value, stage)
    env = {"SWEEP_SPEC": info["spec"], "EXPERIMENT_ID": info["experiment_id"], "RAY_TUNE_JOB_ROOT": info["root"],
           "RESTORE": int((Path(info["root"]) / "experiment.json").exists()),
           "PROFILE_GPUS_PER_NODE": info["gpus_per_node"], "PIPELINE_EXPECTED_NODES": info["nodes"],
           "PIPELINE_CPUS_PER_NODE": info["cpus_per_node"], "PIPELINE_MAX_RESTARTS": value["max_restarts"],
           "PIPELINE_STAGE_COMPLETE": int(complete), "RAY_TUNE_CHECKPOINT_STEPS": value["checkpoint_every_steps"],
           "RAY_TUNE_TRIAL_TIMEOUT": 0}
    record_status(root, stage, "running")
    for key, item in env.items():
        print(f"export {key}={shlex.quote(str(item))}")


def submission_command(value, stage, dependency=None):
    info = value["stages"].get(stage, {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 4})
    exports = f"ALL,PIPELINE_ROOT={value['root']},PIPELINE_STAGE={stage}"
    command = ["sbatch", "--parsable", "--hold", "--partition=gpu", "--account=q1wang", "--qos=normal",
        f"--job-name={value['pipeline_id']}-{stage}", f"--nodes={info['nodes']}", f"--ntasks={info['nodes']}",
        "--ntasks-per-node=1", f"--cpus-per-task={info['cpus_per_node']}", f"--gres=gpu:{info['gpus_per_node']}",
        f"--time={value['walltime']}", "--signal=B:USR1@240", "--requeue", "--open-mode=append",
        f"--output={value['root']}/{stage}-%j.log", f"--export={exports}"]
    if dependency:
        command.extend([f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"])
    return [*command, str(LAUNCHER)]


def queue_pipeline(value, *, resume=False):
    root = Path(value["root"])
    with exclusive(root / ".submission.lock"):
        if (root / "jobs.json").exists() and not resume:
            raise ValueError("Pipeline already submitted; use --resume after a terminal failure")
        if resume and (root / "jobs.json").exists():
            previous = json.loads((root / "jobs.json").read_text())
            ids = [str(job) for job in previous["jobs"].values()]
            live = subprocess.check_output(["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i %T %r"], text=True).splitlines() if ids else []
            states = [line.split(maxsplit=2) for line in live]
            for job, state, reason in states:
                if state != "PENDING" or reason not in {"DependencyNeverSatisfied", "Dependency"}:
                    raise ValueError("Pipeline jobs are still active; automatic requeue owns recovery")
            if states and not any(reason == "DependencyNeverSatisfied" for _, _, reason in states):
                raise ValueError("Pipeline has live dependencies; refusing to replace active work")
            # A dependency that has already failed cannot be repaired by merely
            # requeuing its parent. Replace only the old pending dependent job.
            for line in live:
                subprocess.run(["scancel", line.split()[0]], check=True)
        stages = list(STAGES)
        if resume:
            for stage in STAGES:
                if not (root / f"{stage}_complete.json").exists():
                    break
                if stage in {"ppo", "mappo"}:
                    finish_stage(root, stage)
                else:
                    verify_evaluations(value, stage)
                stages.remove(stage)
            if not stages:
                combine_results(value)
                return
        jobs = {}
        try:
            dependency = None
            for stage in stages:
                output = subprocess.check_output(submission_command(value, stage, dependency), text=True).strip()
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
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, help="Existing pipeline root after terminal failure; retains frozen settings")
    parser.add_argument("--pipeline-id", default="ppo-mappo-mlp108-v1")
    parser.add_argument("--ppo-steps", type=int, default=120000000)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--stage-action", choices=("environment", "finish", "evaluate", "interrupted", "failed"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.stage_action:
        if args.root is None or args.stage is None:
            parser.error("Stage actions require --root and --stage")
        root = args.root.resolve()
        if args.stage_action == "environment":
            stage_environment(root, args.stage)
        elif args.stage_action == "finish":
            finish_stage(root, args.stage)
        elif args.stage_action == "evaluate" and args.stage in {"team", "solo"}:
            evaluate_stage(root, args.stage)
        else:
            load_pipeline(root)
            record_status(root, args.stage, args.stage_action)
        return
    if args.submit and args.dry_run:
        parser.error("Choose --submit or --dry-run")
    if args.resume:
        if not args.submit:
            parser.error("--resume requires --submit")
        if any(flag.split("=", 1)[0] in {"--ppo-steps", "--pipeline-id", "--walltime", "--max-restarts", "--root"} for flag in sys.argv[1:]):
            parser.error("--resume uses frozen settings; omit configuration overrides")
        queue_pipeline(load_pipeline(args.resume.resolve()), resume=True)
        return
    identifier(args.pipeline_id)
    if args.ppo_steps < 1 or args.max_restarts < 0 or not re.fullmatch(r"\d{2,3}:[0-5]\d:[0-5]\d", args.walltime) or args.walltime == "00:00:00":
        parser.error("Positive PPO budget/walltime and nonnegative max-restarts are required")
    root = (args.root or REPO / "outputs/ppo_mappo_pipeline" / args.pipeline_id).expanduser().resolve()
    if any(character in str(root) for character in ",\n"):
        parser.error("Pipeline paths cannot contain commas or newlines (Slurm export syntax)")
    if args.submit:
        if subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() != "multiNode":
            raise ValueError("Expected multiNode branch")
        revision = current_revision()
        value = write_plan(root, args.pipeline_id, args.ppo_steps, args.walltime, args.max_restarts, revision)
        preflight_stage(value, "ppo")
        queue_pipeline(value)
    else:
        with tempfile.TemporaryDirectory(prefix="ppo-mappo-preview-") as temporary:
            value = write_plan(Path(temporary) / "plan", args.pipeline_id, args.ppo_steps, args.walltime, args.max_restarts, "preview")
            preflight_stage(value, "ppo")
            # Use the requested persistent path in the printed submission plan.
            value["root"] = str(root)
            for index, stage in enumerate(STAGES):
                print(shlex.join(submission_command(value, stage, f"{STAGES[index-1].upper()}_JOB_ID" if index else None)))
        print("Preview only: no jobs submitted. PPO -> source verification -> MAPPO -> 6 team finals -> 13 solo finals -> summary.json.")


if __name__ == "__main__":
    main()
