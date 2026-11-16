"""Five-family continual PPO using the existing learner, collector and evaluators."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import time

import torch
import yaml

from agents.common.checkpoints import capture_rng, restore_rng
from core.configuration import CONFIG_ROOT
from core.task_builder import create_race_task
from training.algorithms import check_evaluation_spec
from training.hooks import CheckpointHook, TrainingHook
from training.runtime import preserve_random_state
from training.selection import create_selection_evaluator

FAMILIES = ("straight", "sweeper", "hairpin", "chicane", "s_curve")
SCORES = ("clean_completion_rate", "earned_normalized_progress")


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
    forgetting = [matrix[j + 1][j] - current[j] for j in previous]
    historical = [max(row[j] for row in matrix[j + 1:-1]) - current[j]
                  for j in previous]
    forward = [matrix[j][j] - matrix[0][j]
               for j in range(1, min(learned + 1, len(current)))]
    mean = lambda values: sum(values) / len(values) if values else None
    return dict(average_performance=mean(current), forgetting_mean=mean(forgetting),
                forgetting_max=max(historical) if historical else None,
                historical_forgetting_mean=mean(historical), bwt=mean([-v for v in forgetting]),
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
        if set(config) - {"families", "steps_per_task", "evaluate_every_steps", "thresholds"}:
            raise ValueError("Unsupported continual-learning option")
        self.families = list(config["families"])
        if len(self.families) != 5 or set(self.families) != set(FAMILIES):
            raise ValueError("continual.families must contain each of the five families exactly once")
        for key in ("steps_per_task", "evaluate_every_steps"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"continual.{key} must be a positive integer")
        self.budget, self.cadence = config["steps_per_task"], config["evaluate_every_steps"]
        if scenario["algorithm"]["name"] != "ppo" or scenario["experiment"].get("distributed"):
            raise ValueError("Phase 1 continual learning requires single-process PPO (one GPU per run)")
        if scenario["experiment"].get("total_steps") != self.budget * len(self.families):
            raise ValueError("experiment.total_steps must equal five times continual.steps_per_task")
        if scenario.get("evaluation", {}).get("enabled"):
            raise ValueError("ContinualHook owns evaluation; set evaluation.enabled=false")
        if scenario["adaptation"]["mode"] not in {"full_finetune", "lora"}:
            raise ValueError("Continual PPO requires pretrained actor-and-critic transfer")
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
        self.tasks = []
        for family in self.families:
            maps = yaml.safe_load((CONFIG_ROOT / "maps" / f"generated_{family}.yaml").read_text())
            if len(maps["train"]) != 4 or len(maps["selection"]) != 1:
                raise ValueError(f"Expected four training and one validation map for {family}")
            self.tasks.append(dict(family=family, train=maps["train"], validation=maps["selection"][0]))
        composites = [m for m in scenario["maps"]["selection"] if "/validation_composite_" in m]
        if len(composites) != 3:
            raise ValueError("Continual evaluation requires the three validation composite maps")
        self.validation = {t["family"]: t["validation"] for t in self.tasks}
        self.validation.update({f"composite_{i + 1}": m for i, m in enumerate(composites)})
        if any("/validation_" not in m for m in self.validation.values()):
            raise ValueError("Routine continual evaluation must use validation maps only")
        self.index = self.stage_start = 0
        self.stages, self.evaluations, self.curves = [], [], []
        self.next_evaluation = self.cadence
        self.elapsed = 0.0
        self.update_seconds = self.collection_seconds = 0.0
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
        return {"cl/task_index": self.index + 1 if not self.complete else len(self.tasks),
                "cl/task_family": self.families[self.index] if not self.complete else "complete",
                "cl/task_steps": self._trainer.collected_steps - self.stage_start if not self.complete else 0,
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
                    task_protocol=self.tasks, validation_protocol=self.validation)

    def load_state_dict(self, state):
        if state["task_protocol"] != self.tasks or state["validation_protocol"] != self.validation:
            raise ValueError("Continual map protocol changed since the recovery checkpoint")
        self.index, self.stage_start = state["task_index"], state["stage_start_steps"]
        self.stages, self.evaluations, self.curves = (
            deepcopy(state["completed_stages"]), deepcopy(state["evaluations"]), deepcopy(state["curves"]))
        if (self.index != len(self.stages) or not 0 <= self.index <= len(self.tasks)
                or len(self.evaluations) != self.index + 1
                or state["global_steps"] != self._trainer.collected_steps
                or state["task_steps"] != self._trainer.collected_steps - self.stage_start):
            raise ValueError("Inconsistent continual recovery progress or evaluation matrix")
        self.next_evaluation, self.elapsed = state["next_evaluation"], state["elapsed_seconds"]
        self.update_seconds, self.collection_seconds = state["update_seconds"], state["collection_seconds"]

    def on_update(self, metrics):
        self.update_seconds += metrics.get("perf/update_seconds", 0.0)
        self.collection_seconds += metrics.get("perf/collection_seconds", 0.0)
        metrics.update(self.progress_metrics())

    def on_checkpoint_boundary(self):
        if not self.stage_reached() and self._trainer.collected_steps - self.stage_start >= self.next_evaluation:
            self._record_curve(self._evaluate([self.families[self.index]]))
            steps = self._trainer.collected_steps - self.stage_start
            self.next_evaluation = (steps // self.cadence + 1) * self.cadence
            self._write_results()

    def _evaluate(self, names):
        with preserve_random_state():
            return self._evaluate_maps(names)

    def _evaluate_maps(self, names):
        results = {}
        for name in names:
            evaluation = deepcopy(self.scenario)
            bundle = self.validation[name]
            # A fresh, isolated one-map task guarantees exactly N episodes/map.
            evaluation["maps"]["selection"] = [bundle]
            evaluation["environment"].update(map_bundles=[bundle], map_bundles_train=[bundle],
                map_bundles_eval=[bundle], map_cycle="per_episode", map_pick="round_robin")
            with preserve_random_state():
                evaluator = create_selection_evaluator("ppo", evaluation, self.scenario_dir, self.spec, self.params)
                try:
                    summary = self._trainer.measure_service("evaluation", evaluator.evaluate, self._trainer.agent)
                finally:
                    evaluator.close()
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
            results[name] = summary
            if self.wandb:
                self.wandb.log_metrics({**self.progress_metrics(),
                    "eval/environment_steps": self._trainer.collected_steps, **{
                    f"eval/cl/{name}/{key}": value for key, value in summary.items()
                    if type(value) in (int, float)}})
        return results

    def _record_curve(self, results, *, task_steps=None):
        family = self.families[self.index]
        scores = {key: results[family][key] for key in SCORES}
        self.curves.append(dict(task_index=self.index + 1, family=family,
            task_steps=self._trainer.collected_steps - self.stage_start if task_steps is None else task_steps,
            global_steps=self._trainer.collected_steps, **scores))

    def _matrices(self):
        return {score: [[row["tasks"][family][score] for family in self.families]
                        for row in self.evaluations] for score in SCORES}

    def _aggregate(self):
        return {score: continual_metrics(matrix) if all(v is not None for row in matrix for v in row) else {}
                for score, matrix in self._matrices().items()}

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
                for score in SCORES}
            adaptation.append(dict(family=family, first_observed_threshold_steps=thresholds,
                                   improvement=improvement, observed=bool(curve)))
        seconds = self.wall_seconds()
        sizes = {p.name: p.stat().st_size for p in self.output.glob("*.pt")}
        results = dict(families=self.families, complete=self.complete, completed_stages=self.stages,
            matrices=self._matrices(), metrics=aggregates, adaptation=adaptation, learning_curves=self.curves,
            evaluations=self.evaluations, efficiency=dict(wall_seconds=seconds,
                environment_decisions_per_second=self._trainer.collected_steps / max(seconds, 1e-9),
                physics_steps_per_second=self._trainer.checkpoint_progress()["physics_steps"] / max(seconds, 1e-9),
                update_seconds=self.update_seconds, collection_seconds=self.collection_seconds,
                parameter_counts=self.checkpointer._provenance.get("parameter_counts", {}), checkpoint_bytes=sizes),
            definitions=dict(primary="strict clean finish count / episodes (no contact or boundary excursion)",
                progress="mean of clip(signed earned lap fractions / target_laps, 0, 1) per episode",
                collision="episodes with observed contacts, including nonterminal contacts; collision_dnf_rate is separate",
                forgetting="R[j,j] - R[T,j], j=1..T-1; signed, gains may be negative",
                historical_forgetting="max(R[i,j], i=j..T-1) - R[T,j]",
                fwt="mean(R[j-1,j] - R[0,j]), j=2..5; relative to pretrained PPO",
                adaptation="first observed evaluation crossing; null means not observed, not an imputed budget",
                indexing="row 0 is pretrained; task columns 1..5; interim BWT/forgetting use completed tasks",
                recovery="partial episodes discarded; per-family map/spawn/friction schedules restart at seed"))
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
            baseline = self._evaluate(self.validation)
            self.evaluations.append(dict(completed_stages=0, global_steps=0, tasks=baseline))
            self._record_curve(baseline, task_steps=0)
            trainer.checkpoint_ready = True
            self.checkpointer._save(self.output / "latest.pt", resumable=True)
        self._write_results()  # Replace any output newer than the durable recovery checkpoint.
        try:
            while not self.complete:
                if trainer._should_stop() or trainer._diagnostic_stop_reached():
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
                evaluation = self._evaluate(self.validation)
                self._record_curve(evaluation)
                self.evaluations.append(dict(completed_stages=self.index + 1,
                    global_steps=trainer.collected_steps, tasks=evaluation))
                steps = trainer.collected_steps - self.stage_start
                self.stages.append(dict(family=self.families[self.index], task_index=self.index + 1,
                    requested_steps=self.budget, actual_steps=steps, overshoot=steps - self.budget,
                    global_steps=trainer.collected_steps, partial_episodes="bootstrapped in PPO, discarded on collector switch",
                    performance=trainer.performance_metrics()))
                self.index += 1
                self.stage_start, self.next_evaluation = trainer.collected_steps, self.cadence
                if not self.complete:
                    self._record_curve(evaluation, task_steps=0)
                # Metadata commits the evaluation and next stage in the same atomic payload as optimizer/RNG.
                self.checkpointer._save(self.output / "latest.pt", resumable=True)
                self.checkpointer._save(self.output / f"task_{self.index}_{self.families[self.index - 1]}.pt", resumable=True)
                self._write_results()
            for hook in trainer.hooks:
                trainer.call_hook(hook, "on_training_end")
        finally:
            self.elapsed = self.wall_seconds()
            self._started = None
        self._write_results()
