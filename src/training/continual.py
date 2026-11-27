"""Continual PPO using the existing learner, collector and evaluators."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import time

import torch
import yaml

from agents.common.checkpoints import capture_rng, checkpoint_sha256, restore_rng
from core.configuration import CONFIG_ROOT, source_path
from core.map_selection import resolve_bundle_yaml
from core.provenance import collect_map_protocols
from core.task_builder import create_race_task
from training.algorithms import check_evaluation_spec
from training.hooks import CheckpointHook, TrainingHook
from training.runtime import preserve_random_state
from training.selection import BackgroundSelectionEvaluator, _evaluate_and_save, create_selection_evaluator

FAMILIES = ("straight", "sweeper", "hairpin", "chicane", "s_curve")
SCORES = ("clean_completion_rate", "earned_normalized_progress")
HIERARCHICAL_SCORES = (*SCORES, "mean_net_progress", "mean_return_ll_undiscounted", "mean_speed")


def map_tasks(config, scenario):
    """Resolve explicit ordered tasks without copying either map dataset."""
    entries = config["tasks"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("continual.tasks must be a nonempty ordered list")
    manifest_path = config.get("generated_manifest")
    manifest = None
    if manifest_path:
        manifest_path = source_path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
    tasks = []
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) - {"name", "map", "generated_slot", "domain", "complexity"}
                or not isinstance(entry.get("name"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["name"])
                or not isinstance(entry.get("domain"), str) or not entry["domain"]
                or ("map" in entry) == ("generated_slot" in entry)):
            raise ValueError("Each continual task needs a name, domain and exactly one map or generated_slot")
        if "generated_slot" in entry:
            if manifest is None:
                raise ValueError("generated_slot requires continual.generated_manifest")
            matches = [row for row in manifest["accepted"] if row["slot_id"] == entry["generated_slot"]]
            if len(matches) != 1 or matches[0]["split"] != "train":
                raise ValueError(f"Generated task must identify exactly one training map: {entry['generated_slot']}")
            selected = matches[0]
            if "complexity" in entry and selected.get("complexity") != entry["complexity"]:
                raise ValueError(f"Generated task complexity differs from manifest: {entry['name']}")
            bundle = (manifest_path.parent / selected["map"]).resolve()
            configured_train = {resolve_bundle_yaml(Path(scenario["environment"]["map_dir"]), item)
                                for item in scenario["maps"]["train"]}
            if bundle not in configured_train:
                raise ValueError(f"Generated task is not in the configured training split: {entry['name']}")
        else:
            bundle = source_path(entry["map"])
            if manifest is not None and bundle.is_relative_to(manifest_path.parent):
                matches = [row for row in manifest["accepted"]
                           if (manifest_path.parent / row["map"]).resolve() == bundle]
                if len(matches) != 1 or matches[0]["split"] != "train":
                    raise ValueError(f"Generated task is not a manifest training map: {entry['name']}")
                configured_train = {resolve_bundle_yaml(Path(scenario["environment"]["map_dir"]), item)
                                    for item in scenario["maps"]["train"]}
                if bundle not in configured_train:
                    raise ValueError(f"Generated task is not in the configured training split: {entry['name']}")
        if not bundle.is_file():
            raise FileNotFoundError(f"Continual task map not found: {bundle}")
        tasks.append(dict(family=entry["name"], train=[str(bundle)], validation=str(bundle),
                          domain=entry["domain"], **({"complexity": entry["complexity"]}
                          if "complexity" in entry else {})))
    if len({task["family"] for task in tasks}) != len(tasks):
        raise ValueError("Continual task names must be distinct")
    return tasks


def continual_metrics(matrix):
    """Rows are baseline, after task 1, ...; columns follow task order.

    FWT = mean(R[j-1,j] - R[0,j]), j=2..last observed upcoming task.
    Historical forgetting uses post-acquisition boundary evaluations only.
    """
    learned = len(matrix) - 1
    if not matrix:
        return {}
    current = matrix[-1]
    previous = range(max(0, learned - 1))
    difference = lambda left, right: left - right if left is not None and right is not None else None
    forgetting = [difference(matrix[j + 1][j], current[j]) for j in previous]
    historical = []
    for j in previous:
        values = [row[j] for row in matrix[j + 1:-1]]
        peak = max(values) if values and all(v is not None for v in values) else None
        historical.append(difference(peak, current[j]))
    forward = [difference(matrix[j][j], matrix[0][j])
               for j in range(1, min(learned + 1, len(current)))]
    mean = lambda values: sum(values) / len(values) if values and all(v is not None for v in values) else None
    return dict(average_performance=mean(current), forgetting_mean=mean(forgetting),
                forgetting_max=max(historical) if historical and all(v is not None for v in historical) else None,
                historical_forgetting_mean=mean(historical), bwt=mean([-v if v is not None else None for v in forgetting]),
                fwt=mean(forward), forgetting_per_task=forgetting,
                historical_forgetting_per_task=historical)


def _atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ContinualHook(TrainingHook):
    """Own stage/evaluation metadata, never model or optimizer construction."""
    service_name = "logging"
    requires_transition_record = False

    def __init__(self, scenario, scenario_dir, spec, params, output_dir, *, wandb=None):
        self.scenario, self.scenario_dir = deepcopy(scenario), scenario_dir
        self.spec, self.params, self.wandb = spec, params, wandb
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        config = scenario["continual"]
        if set(config) - {"families", "tasks", "generated_manifest", "steps_per_task",
                          "evaluate_every_steps", "thresholds", "stop_after_task"}:
            raise ValueError("Unsupported continual-learning option")
        if ("families" in config) == ("tasks" in config):
            raise ValueError("Configure exactly one of continual.families and continual.tasks")
        self.explicit_tasks = "tasks" in config
        self.tasks = []
        if self.explicit_tasks:
            self.tasks = map_tasks(config, scenario)
            self.families = [task["family"] for task in self.tasks]
        else:
            self.families = list(config["families"])
            if (not self.families or len(set(self.families)) != len(self.families)
                    or not set(self.families) <= set(FAMILIES)):
                raise ValueError("continual.families must contain distinct known families")
        stop_override = os.environ.get("CONTINUAL_STOP_AFTER_TASK")
        if stop_override is not None and not stop_override.isdecimal():
            raise ValueError("CONTINUAL_STOP_AFTER_TASK must be a positive task number")
        self.stop_after_task = int(stop_override) if stop_override is not None else config.get("stop_after_task")
        if self.stop_after_task is not None and (type(self.stop_after_task) is not int
                or not 1 <= self.stop_after_task <= len(self.families)):
            raise ValueError("continual.stop_after_task must be null or a task number")
        for key in ("steps_per_task", "evaluate_every_steps"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"continual.{key} must be a positive integer")
        self.budget, self.cadence = config["steps_per_task"], config["evaluate_every_steps"]
        if scenario["algorithm"]["name"] != "ppo" or scenario["experiment"].get("distributed"):
            raise ValueError("Phase 1 continual learning requires single-process PPO (one GPU per run)")
        if scenario["experiment"].get("total_steps") != self.budget * len(self.families):
            raise ValueError("experiment.total_steps must equal the task count times continual.steps_per_task")
        if scenario.get("evaluation", {}).get("enabled"):
            raise ValueError("ContinualHook owns evaluation; set evaluation.enabled=false")
        hierarchical = scenario.get("hierarchical", {}).get("enabled", False)
        if scenario["adaptation"]["mode"] not in ({"scratch"} if hierarchical else {"full_finetune", "lora"}):
            raise ValueError("Continual PPO requires a fresh hierarchical selector or pretrained conventional PPO")
        self.scores = HIERARCHICAL_SCORES if hierarchical else SCORES
        lora = params.get("lora")
        if lora and (lora["rank"] != 4 or lora["mode"] != "shared"):
            raise ValueError("Phase 1 requires one shared rank-4 adapter")
        self.thresholds = config.get("thresholds", {})
        if set(self.thresholds) - set(SCORES):
            raise ValueError(f"continual.thresholds accepts {SCORES}")
        for values in self.thresholds.values():
            if not isinstance(values, list) or any(type(v) not in (int, float)
                    or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Adaptation thresholds must be lists of finite numbers in [0,1]")
        if not self.explicit_tasks:
            for family in self.families:
                maps = yaml.safe_load((CONFIG_ROOT / "maps" / f"generated_{family}.yaml").read_text())
                if len(maps["train"]) != 4 or len(maps["selection"]) != 1:
                    raise ValueError(f"Expected four training and one validation map for {family}")
                self.tasks.append(dict(family=family, train=maps["train"], validation=maps["selection"][0]))
        composites = (list(scenario["maps"]["selection"]) if self.explicit_tasks else
                      [m for m in scenario["maps"]["selection"] if "/validation_composite_" in m])
        if not self.explicit_tasks and len(composites) != 3:
            raise ValueError("Continual evaluation requires the three validation composite maps")
        if self.explicit_tasks and set(composites) & set(scenario["maps"]["final_test"]):
            raise ValueError("Routine continual validation cannot include final-test maps")
        self.validation = {t["family"]: t["validation"] for t in self.tasks}
        prefix = "validation" if self.explicit_tasks else "composite"
        if self.explicit_tasks and any(f"{prefix}_{i + 1}" in self.validation for i in range(len(composites))):
            raise ValueError("Continual task names conflict with generated validation labels")
        self.validation.update({f"{prefix}_{i + 1}": m for i, m in enumerate(composites)})
        if not self.explicit_tasks and any("/validation_" not in m for m in self.validation.values()):
            raise ValueError("Routine continual evaluation must use validation maps only")
        self.composites = [f"{prefix}_{i + 1}" for i in range(len(composites))]
        self.map_provenance = None
        if self.explicit_tasks:
            environment = deepcopy(scenario["environment"])
            environment["map_bundles"] = [bundle for task in self.tasks for bundle in task["train"]]
            environment["map_bundles_train"] = environment["map_bundles"]
            environment["map_bundles_eval"] = list(self.validation.values())
            self.map_provenance = collect_map_protocols(environment)
        self.index = self.stage_start = 0
        self.stage_start_counters = dict(agent_steps=0, updates=0, optimizer_steps=0)
        self.stages, self.evaluations, self.curves = [], [], []
        self.next_evaluation = self.cadence
        self.elapsed = 0.0
        self.update_seconds = self.collection_seconds = 0.0
        self.pending_evaluations = []
        self._background = None
        self._started = None

    @property
    def complete(self):
        return self.index == len(self.tasks)

    def bind_trainer(self, trainer):
        super().bind_trainer(trainer)
        trainer.continual = self
        self.checkpointer = next(h for h in trainer.hooks if type(h) is CheckpointHook)

    def stage_reached(self):
        return self.complete or self._trainer.collected_steps - self.stage_start >= self.budget

    def progress_metrics(self):
        progress = self._trainer.checkpoint_progress()
        return {"cl/task_index": self.index + 1 if not self.complete else len(self.tasks),
                "cl/task_family": self.families[self.index] if not self.complete else "complete",
                "cl/task_steps": self._trainer.collected_steps - self.stage_start if not self.complete else 0,
                **({f"cl/task_{key}": progress[key] - self.stage_start_counters[key]
                    for key in self.stage_start_counters} if self.explicit_tasks else {}),
                "cl/global_steps": self._trainer.collected_steps}

    def wall_seconds(self):
        return self.elapsed + (time.perf_counter() - self._started if self._started is not None else 0.0)

    def state_dict(self):
        return dict(task_index=self.index, task_family=self.families[self.index] if not self.complete else None,
                    completed_stages=self.stages, stage_start_steps=self.stage_start,
                    task_steps=self._trainer.collected_steps - self.stage_start,
                    global_steps=self._trainer.collected_steps, next_evaluation=self.next_evaluation,
                    evaluations=self.evaluations, curves=self.curves, elapsed_seconds=self.wall_seconds(),
                    update_seconds=self.update_seconds, collection_seconds=self.collection_seconds,
                    task_protocol=self.tasks, validation_protocol=self.validation,
                    **({"task_map_provenance": self.map_provenance,
                        "pending_evaluations": self.pending_evaluations,
                        "stage_start_counters": self.stage_start_counters} if self.explicit_tasks else {}))

    def load_state_dict(self, state):
        if state["task_protocol"] != self.tasks or state["validation_protocol"] != self.validation:
            raise ValueError("Continual map protocol changed since the recovery checkpoint")
        if self.explicit_tasks and state.get("task_map_provenance") != self.map_provenance:
            raise ValueError("Continual task map assets changed since the recovery checkpoint")
        self.index, self.stage_start = state["task_index"], state["stage_start_steps"]
        self.stages, self.evaluations, self.curves = (
            deepcopy(state["completed_stages"]), deepcopy(state["evaluations"]), deepcopy(state["curves"]))
        if (self.index != len(self.stages) or not 0 <= self.index <= len(self.tasks)
                or not (len(self.evaluations) == self.index + 1
                        or self.index == 0 and not self.evaluations)
                or state["global_steps"] != self._trainer.collected_steps
                or state["task_steps"] != self._trainer.collected_steps - self.stage_start):
            raise ValueError("Inconsistent continual recovery progress or evaluation matrix")
        self.next_evaluation, self.elapsed = state["next_evaluation"], state["elapsed_seconds"]
        self.update_seconds, self.collection_seconds = state["update_seconds"], state["collection_seconds"]
        self.pending_evaluations = deepcopy(state.get("pending_evaluations", []))
        self.stage_start_counters = deepcopy(state.get("stage_start_counters", self.stage_start_counters))
        if self.explicit_tasks:
            progress = self._trainer.checkpoint_progress()
            if (set(self.stage_start_counters) != {"agent_steps", "updates", "optimizer_steps"}
                    or any(type(value) is not int or not 0 <= value <= progress[key]
                           for key, value in self.stage_start_counters.items())):
                raise ValueError("Invalid continual task-start counters")
        for job in self.pending_evaluations:
            if not self._snapshot_path(job["policy_version"]).is_file():
                raise ValueError("Missing continual evaluation snapshot required for recovery")

    def on_update(self, metrics):
        self.update_seconds += metrics.get("perf/update_seconds", 0.0)
        self.collection_seconds += metrics.get("perf/collection_seconds", 0.0)
        metrics.update(self.progress_metrics())

    def on_checkpoint_boundary(self):
        if self.explicit_tasks:
            self._collect_pending()
        if not self.stage_reached() and self._trainer.collected_steps - self.stage_start >= self.next_evaluation:
            if self.explicit_tasks:
                self._queue_evaluation()
            else:
                self._record_curve(self._evaluate([self.families[self.index]]))
            steps = self._trainer.collected_steps - self.stage_start
            self.next_evaluation = (steps // self.cadence + 1) * self.cadence
            self._write_results()

    def _snapshot_path(self, version):
        return self.output / ".continual_evaluation" / f"{version:09d}.pt"

    def _evaluation_scenario(self, name, episodes):
        evaluation = deepcopy(self.scenario)
        bundle = self.validation[name]
        evaluation["maps"]["selection"] = [bundle]
        evaluation["evaluation"]["episodes"] = episodes
        evaluation["environment"].update(map_bundles=[bundle], map_bundles_train=[bundle],
            map_bundles_eval=[bundle], map_cycle="per_episode", map_pick="round_robin")
        return evaluation

    def _start_pending(self):
        if self.pending_evaluations and self._background is None:
            job = self.pending_evaluations[0]
            self._background = BackgroundSelectionEvaluator("ppo", self._evaluation_scenario(job["name"], 5),
                self.scenario_dir, self.spec, self.params, self.output / ".continual_evaluation", device="cpu")
            self._background.start(job["policy_version"])

    def _queue_evaluation(self):
        if len(self.pending_evaluations) >= 2:
            self._collect_pending(wait=True)
        progress = self._trainer.checkpoint_progress()
        version = progress["updates"]
        snapshot = self._snapshot_path(version)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer._save(snapshot)
        snapshot.with_suffix(".json").unlink(missing_ok=True)
        self.pending_evaluations.append(dict(name=self.families[self.index], task_index=self.index + 1,
            task_steps=self._trainer.collected_steps - self.stage_start, global_steps=self._trainer.collected_steps,
            policy_version=version, snapshot_sha256=checkpoint_sha256(snapshot), queued_unix=time.time()))
        self.checkpointer.request_recovery_checkpoint()
        self._start_pending()

    def _collect_pending(self, *, wait=False):
        self._start_pending()
        if self._background is None:
            return
        job = self.pending_evaluations[0]
        result = (self._trainer.measure_service("evaluation", self._background.result,
                                                job["policy_version"], wait=True)
                  if wait else self._background.result(job["policy_version"], wait=False))
        if result is None:
            return
        summary, seconds = result
        if summary["snapshot_sha256"] != job["snapshot_sha256"]:
            raise ValueError("Continual evaluation result does not match its policy snapshot")
        summary.update(evaluation_seconds=seconds,
                       evaluation_lag_steps=self._trainer.collected_steps - job["global_steps"],
                       evaluation_queue_wait_seconds=max(0., summary["evaluation_started_unix"] - job["queued_unix"]))
        summary = self._summarize(job["name"], summary, job)
        self._record_curve({job["name"]: summary}, task_steps=job["task_steps"],
                           task_index=job["task_index"], global_steps=job["global_steps"],
                           policy_version=job["policy_version"])
        self.curves[-1]["evaluation"] = summary
        self.pending_evaluations.pop(0)
        self.checkpointer.request_recovery_checkpoint()
        self._background.close()
        self._background = None
        self._start_pending()
        self._write_results()

    def _evaluate(self, names, *, episodes=None):
        with preserve_random_state():
            if self.explicit_tasks:
                return self._trainer.measure_service("evaluation", self._evaluate_parallel,
                                                     list(names), episodes=episodes)
            return self._evaluate_maps(names, episodes=episodes)

    def _evaluate_parallel(self, names, *, episodes):
        # Collectors have stopped at task boundaries. The existing snapshot
        # evaluator runs independent maps on CPU, bounded by allocated affinity.
        workers = min(len(names), 24, max(1, len(os.sched_getaffinity(0)) - 2))
        with tempfile.TemporaryDirectory(prefix=".cl-boundary-", dir=self.output) as directory:
            directory = Path(directory)
            snapshot = directory / "snapshot.pt"
            self.checkpointer._save(snapshot)
            params = {**self.params, "device": "cpu", "compile_networks": False}
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                jobs = []
                for index, name in enumerate(names):
                    path = directory / f"result_{index}.json"
                    job = pool.submit(_evaluate_and_save, "ppo", self._evaluation_scenario(name, episodes),
                                      self.scenario_dir, self.spec, params, snapshot, path)
                    jobs.append((name, path, job))
                results = {}
                for name, path, job in jobs:
                    job.result()
                    summary, seconds = json.loads(path.read_text())
                    summary["evaluation_seconds"] = seconds
                    results[name] = self._summarize(name, summary)
                return results

    def _evaluate_maps(self, names, *, episodes=None):
        results = {}
        for name in names:
            evaluation = self._evaluation_scenario(name, episodes or self.scenario["evaluation"]["episodes"])
            with preserve_random_state():
                evaluator = create_selection_evaluator("ppo", evaluation, self.scenario_dir, self.spec, self.params)
                try:
                    summary = self._trainer.measure_service("evaluation", evaluator.evaluate, self._trainer.agent)
                finally:
                    evaluator.close()
            results[name] = self._summarize(name, summary)
        return results

    def _summarize(self, name, summary, job=None):
        records = summary["episode_results"]
        progress = [row["mean_net_progress_laps"] for row in records]
        laps = summary["evaluation_protocol"]["target_laps"]
        reasons = [agent["terminal_reason"] for row in records for agent in row["agents"].values()
                   if agent["team"] == "trainable"]
        summary.update(clean_completion_rate=summary["strict_clean_finish_count"] / summary["episodes"],
            earned_normalized_progress=(sum(max(0., min(1., value / laps)) for value in progress)
                / len(progress) if all(value is not None for value in progress) else None),
            collision_dnf_rate=summary["collision_rate"],
            collision_rate=sum(row["collision_observed"] for row in records) / len(records),
            track_boundary_violation_rate=sum(row["boundary_violation_observed"] for row in records) / len(records),
            timeout_termination_rate=reasons.count("time_limit") / len(reasons),
            no_progress_termination_rate=reasons.count("no_progress") / len(reasons))
        hierarchy = [row["hierarchical"] for row in records if "hierarchical" in row]
        if hierarchy:
            names = [entry["name"] for entry in hierarchy[0]["skills"]]
            composition = 'composition/dominant_selection_count/' + names[0] in hierarchy[0]['metrics']
            prefix = 'composition/dominant_' if composition else 'hierarchical/'
            counts = {skill: sum(row["metrics"][f"{prefix}selection_count/{skill}"]
                                 for row in hierarchy) for skill in names}
            total = sum(counts.values())
            proportions = {skill: count / total if total else 0.0 for skill, count in counts.items()}
            transitions = [[sum(row["transition_counts"][i][j] for row in hierarchy)
                            for j in range(len(names))] for i in range(len(names))]
            summary["hierarchical_diagnostics"] = dict(selection_counts=counts,
                selection_proportions=proportions,
                selection_entropy=-sum(p * math.log(p) for p in proportions.values() if p),
                transitions=transitions, skill_order=names,
                mean_dwell_ll_decisions=sum(row["metrics"][f"{prefix}mean_dwell_ll_decisions"]
                                            for row in hierarchy) / len(hierarchy))
            if composition:
                summary['hierarchical_diagnostics']['mean_dominant_transitions'] = sum(
                    row['metrics']['composition/dominant_transitions'] for row in hierarchy) / len(hierarchy)
                keys = [key for key in hierarchy[0]['metrics'] if key.startswith('composition/')]
                summary['composition_diagnostics'] = {key.removeprefix('composition/'): sum(
                    row['metrics'][key] for row in hierarchy) / len(hierarchy) for key in keys}
                summary['hierarchical_diagnostics']['dominant_option_counts'] = summary['hierarchical_diagnostics'].pop('selection_counts')
                summary['hierarchical_diagnostics']['dominant_option_proportions'] = summary['hierarchical_diagnostics'].pop('selection_proportions')
                summary['hierarchical_diagnostics']['dominant_option_entropy'] = summary['hierarchical_diagnostics'].pop('selection_entropy')
            else:
                summary['hierarchical_diagnostics']['mean_switches'] = sum(
                    row['metrics']['hierarchical/switches'] for row in hierarchy) / len(hierarchy)
        job = job or {"policy_version": self._trainer.checkpoint_progress()["updates"],
                      "task_index": self.index + 1 if not self.complete else len(self.tasks),
                      "global_steps": self._trainer.collected_steps}
        summary.update(policy_version=job["policy_version"], task_index=job["task_index"],
                       ll_decisions=job["global_steps"])
        if self.wandb:
            diagnostics = summary.get("hierarchical_diagnostics", {})
            self.wandb.log_metrics({"cl/task_index": job["task_index"],
                "eval/environment_steps": job["global_steps"], **{
                f"eval/cl/{name}/{key}": value for key, value in summary.items()
                if type(value) in (int, float)}, **{
                f"eval/cl/{name}/skill_proportion/{skill}": value
                for skill, value in diagnostics.get("selection_proportions", {}).items()}, **{
                f"eval/cl/{name}/hierarchical/{key}": value for key, value in diagnostics.items()
                if type(value) in (int, float)}, **{
                f"eval/cl/{name}/composition/{key}": value for key, value in summary.get('composition_diagnostics', {}).items()})
        return summary

    def _record_curve(self, results, *, task_steps=None, task_index=None, global_steps=None, policy_version=None):
        task_index = self.index + 1 if task_index is None else task_index
        family = self.families[task_index - 1]
        scores = {key: results[family].get(key) for key in self.scores}
        self.curves.append(dict(task_index=task_index, family=family,
            task_steps=self._trainer.collected_steps - self.stage_start if task_steps is None else task_steps,
            global_steps=self._trainer.collected_steps if global_steps is None else global_steps,
            **({"policy_version": policy_version} if policy_version is not None else {}), **scores))

    def _matrices(self):
        return {score: [[row["tasks"][family].get(score) for family in self.families]
                        for row in self.evaluations] for score in self.scores}

    def _aggregate(self):
        return {score: continual_metrics(matrix) for score, matrix in self._matrices().items()}

    def _write_results(self):
        aggregates = self._aggregate()
        adaptation = []
        for index, family in enumerate(self.families, 1):
            curve = [p for p in self.curves if p["task_index"] == index]
            thresholds = {score: {str(threshold): next((p["task_steps"] for p in curve
                if p[score] is not None and p[score] >= threshold), None)
                for threshold in values} for score, values in self.thresholds.items()}
            improvement = {score: curve[-1][score] - curve[0][score]
                if curve and curve[-1][score] is not None and curve[0][score] is not None else None
                for score in self.scores}
            jumpstart = {score: curve[0][score] if curve else None for score in self.scores}
            adaptation.append(dict(family=family, jumpstart=jumpstart, first_observed_threshold_steps=thresholds,
                                   improvement=improvement, observed=bool(curve)))
        seconds = self.wall_seconds()
        progress = self._trainer.checkpoint_progress()
        training_seconds = self.update_seconds + self.collection_seconds
        sizes = {p.name: p.stat().st_size for p in self.output.glob("*.pt")}
        results = dict(families=self.families, task_protocol=self.tasks,
            complete=self.complete, completed_stages=self.stages,
            matrices=self._matrices(), metrics=aggregates, adaptation=adaptation, learning_curves=self.curves,
            evaluations=self.evaluations, efficiency=dict(wall_seconds=seconds,
                environment_decisions_per_second=self._trainer.collected_steps / max(seconds, 1e-9),
                physics_steps_per_second=self._trainer.checkpoint_progress()["physics_steps"] / max(seconds, 1e-9),
                training_seconds=training_seconds,
                training_ll_decisions_per_second=(progress["environment_steps"] / training_seconds
                                                  if training_seconds else None),
                training_hl_decisions_per_second=(progress["agent_steps"] / training_seconds
                                                  if training_seconds else None),
                training_optimizer_steps_per_second=(progress["optimizer_steps"] / training_seconds
                                                     if training_seconds else None),
                services=self._trainer.service_metrics(),
                update_seconds=self.update_seconds, collection_seconds=self.collection_seconds,
                parameter_counts=self.checkpointer._provenance.get("parameter_counts", {}), checkpoint_bytes=sizes),
            definitions=dict(primary="strict clean finish count / episodes (no contact or boundary excursion)",
                progress="mean of clip(signed earned lap fractions / target_laps, 0, 1) per episode",
                collision="episodes with observed contacts, including nonterminal contacts; collision_dnf_rate is separate",
                forgetting="R[j,j] - R[T,j], j=1..T-1; signed, gains may be negative",
                historical_forgetting="max(R[i,j], i=j..T-1) - R[T,j]",
                fwt="mean(R[j-1,j] - R[0,j]) for observed upcoming tasks, relative to the initial policy",
                adaptation="first observed evaluation crossing; null means not observed, not an imputed budget",
                indexing="row 0 is initial policy; task columns follow the configured order; interim BWT/forgetting use completed tasks",
                recovery="partial episodes discarded; per-family map/spawn/friction schedules restart at seed"))
        if self.explicit_tasks:
            domains = sorted({task["domain"] for task in self.tasks})
            results["domain_transitions"] = [dict(after_task=i, source=self.tasks[i - 1]["domain"],
                target=self.tasks[i]["domain"]) for i in range(1, len(self.tasks))
                if self.tasks[i - 1]["domain"] != self.tasks[i]["domain"]]
            results["domain_performance"] = {}
            for domain in domains:
                columns = [i for i, task in enumerate(self.tasks) if task["domain"] == domain]
                results["domain_performance"][domain] = {}
                for score, matrix in self._matrices().items():
                    rows = []
                    for row in matrix:
                        values = [row[i] for i in columns]
                        rows.append(sum(values) / len(values) if all(v is not None for v in values) else None)
                    results["domain_performance"][domain][score] = rows
        _atomic_json(self.output / "continual_results.json", results)
        for score, matrix in self._matrices().items():
            path = self.output / f"continual_{score}.csv"
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["completed_stages", *self.families])
                writer.writerows([i, *row] for i, row in enumerate(matrix))
            temporary.replace(path)
        if self.wandb:
            self.wandb.log_metrics({**self.progress_metrics(), **{f"cl/{key}": value
                for key, value in aggregates[SCORES[0]].items() if type(value) in (int, float)},
                **{f"cl/domain/{domain}/{score}": rows[-1]
                   for domain, scores in results.get("domain_performance", {}).items()
                   for score, rows in scores.items() if rows and rows[-1] is not None},
                "train/environment_steps": self._trainer.collected_steps,
                "perf/cl_wall_seconds": seconds,
                "perf/cl_environment_decisions_per_second": self._trainer.collected_steps / max(seconds, 1e-9),
                "perf/cl_physics_steps_per_second": self._trainer.checkpoint_progress()["physics_steps"] / max(seconds, 1e-9),
                **{f"perf/cl_parameters/{key}": value for key, value in
                   self.checkpointer._provenance.get("parameter_counts", {}).items() if type(value) is int},
                **{f"perf/cl_checkpoint_bytes/{key}": value for key, value in sizes.items()}})

    def run(self, trainer):
        """Rebuild collectors only; retain the learner, hooks and global counters."""
        self.elapsed += trainer.initialization_seconds
        self._started = time.perf_counter()
        total_steps = self.budget * len(self.tasks)
        trainer.total_steps, trainer.n_episodes = total_steps, 0
        if trainer._resume_state is not None:
            restore_rng(trainer._resume_state["rng"])
            trainer.checkpoint_ready = True
        if not self.evaluations:
            # Make baseline evaluation recoverable even if the allocation ends
            # before its initial CPU evaluations finish.
            trainer.checkpoint_ready = True
            self.checkpointer._save(self.output / "latest.pt", resumable=True)
            baseline = self._evaluate(self.families if self.explicit_tasks else self.validation,
                                      episodes=10 if self.explicit_tasks else None)
            if self.explicit_tasks:
                baseline.update(self._evaluate(self.composites, episodes=5))
            self.evaluations.append(dict(completed_stages=0, global_steps=0, tasks=baseline))
            self._record_curve(baseline, task_steps=0)
            trainer.checkpoint_ready = True
            self.checkpointer._save(self.output / "latest.pt", resumable=True)
        self._write_results()  # Replace any output newer than the durable recovery checkpoint.
        try:
            if self.explicit_tasks:
                self._start_pending()
            while not self.complete:
                if (trainer._should_stop() or trainer._diagnostic_stop_reached()
                        or self.stop_after_task is not None and self.index >= self.stop_after_task):
                    break
                if not self.stage_reached():
                    active = deepcopy(self.scenario)
                    maps = self.tasks[self.index]["train"]
                    active["environment"].update(map_bundles=maps, map_bundles_train=maps,
                        map_bundles_eval=maps, map_cycle="per_episode", map_pick="round_robin")
                    trainer._collection_rng = capture_rng()
                    with preserve_random_state():
                        task = create_race_task(active, scenario_dir=self.scenario_dir)
                    threads = torch.get_num_threads()
                    try:
                        check_evaluation_spec(self.spec, task.spec)
                        trainer.task, trainer.scenario = task, active
                        trainer.reports.task = task
                        # Worker trajectory IDs and local episode counters restart with collectors.
                        trainer._worker_policy_starts.clear()
                        trainer.train(total_steps=total_steps, finalize=False)
                    finally:
                        task.close()
                        # Process Collector shutdown can increase the parent's
                        # thread pool; do not accumulate that increase per family.
                        torch.set_num_threads(threads)
                    trainer._resume_state = None
                if not self.stage_reached():
                    break
                if not trainer.checkpoint_ready or trainer._pending_learning_steps:
                    raise RuntimeError("Task transition requires a completed PPO update and empty rollouts")
                while self.pending_evaluations:
                    self._collect_pending(wait=True)
                transition_started = time.perf_counter()
                evaluation = self._evaluate(self.families if self.explicit_tasks else self.validation,
                                            episodes=10 if self.explicit_tasks else None)
                if self.explicit_tasks and (self.index + 1 == len(self.tasks)
                        or self.tasks[self.index]["domain"] != self.tasks[self.index + 1]["domain"]):
                    evaluation.update(self._evaluate(self.composites, episodes=5))
                self._record_curve(evaluation)
                self.evaluations.append(dict(completed_stages=self.index + 1,
                    global_steps=trainer.collected_steps, tasks=evaluation))
                steps = trainer.collected_steps - self.stage_start
                progress = trainer.checkpoint_progress()
                self.stages.append(dict(family=self.families[self.index], task_index=self.index + 1,
                    requested_steps=self.budget, actual_steps=steps, overshoot=steps - self.budget,
                    global_steps=trainer.collected_steps, partial_episodes="bootstrapped in PPO, discarded on collector switch",
                    performance=trainer.performance_metrics(),
                    **({"domain": self.tasks[self.index]["domain"],
                        "counters": {key: progress[key] - self.stage_start_counters[key]
                                     for key in self.stage_start_counters},
                        "domain_transition_from": (self.tasks[self.index - 1]["domain"]
                            if self.index and self.tasks[self.index - 1]["domain"] != self.tasks[self.index]["domain"]
                            else None),
                        "boundary_evaluation_seconds": time.perf_counter() - transition_started}
                       if self.explicit_tasks else {})))
                self.index += 1
                self.stage_start, self.next_evaluation = trainer.collected_steps, self.cadence
                self.stage_start_counters = {key: progress[key] for key in self.stage_start_counters}
                if not self.complete:
                    self._record_curve(evaluation, task_steps=0)
                # Metadata commits the evaluation and next stage in the same atomic payload as optimizer/RNG.
                self.checkpointer._save(self.output / "latest.pt", resumable=True)
                self.checkpointer._save(self.output / f"task_{self.index}_{self.families[self.index - 1]}.pt", resumable=True)
                self._write_results()
            for hook in trainer.hooks:
                trainer.call_hook(hook, "on_training_end")
        finally:
            if self._background is not None:
                self._background.close()
                self._background = None
            self.elapsed = self.wall_seconds()
            self._started = None
        self._write_results()
