"""Build isolated checkpoint selection tasks without disturbing training RNGs."""
from copy import deepcopy
import json
from multiprocessing import get_context
import os
from pathlib import Path
import resource
import time

from adapters.rewards import RewardMapping
from training.evaluation_config import resolve_evaluation_protocol
from core.task_builder import create_race_task
from training.algorithms import check_evaluation_spec
from training.runtime import preserve_random_state, seed_process


def create_selection_evaluator(algorithm, scenario, scenario_dir, training_spec, params):
    protocol = resolve_evaluation_protocol(scenario, "selection")
    evaluation = deepcopy(scenario)
    evaluation["experiment"]["seed"] = protocol["seed"]
    evaluation["environment"]["max_steps"] = protocol["max_steps"]
    evaluation.setdefault("evaluation", {})["max_steps"] = protocol["max_steps"]
    if "target_laps" in protocol:
        evaluation["evaluation"]["target_laps"] = protocol["target_laps"]
    with preserve_random_state():
        seed_process(protocol["seed"])
        task = create_race_task(evaluation, scenario_dir=scenario_dir, mode="eval")
    try:
        check_evaluation_spec(training_spec, task.spec)
        options = dict(task=task, episodes=protocol["episodes"], base_seed=protocol["seed"],
                       reward_mapping=RewardMapping(params.get("reward_mode", "individual"),
                                                    params.get("team_reward_reduction", "mean")))
        if algorithm == "ppo":
            from training.ppo_evaluator import DeterministicPPOEvaluator
            return DeterministicPPOEvaluator(**options)
        if algorithm != "mappo":
            raise ValueError(f"Unsupported evaluation algorithm: {algorithm!r}")
        from training.mappo_evaluator import DeterministicMAPPOEvaluator
        return DeterministicMAPPOEvaluator(**options)
    except BaseException:
        task.close()
        raise


def _evaluate_selection_checkpoint(algorithm, scenario, scenario_dir, spec, params, checkpoint, result_path):
    """Run the selection protocol on a snapshot in a spawned process."""
    import torch
    from agents.common.checkpoints import read_checkpoint
    from training.algorithms import create_learner

    torch.set_num_threads(1)
    evaluator = create_selection_evaluator(algorithm, scenario, scenario_dir, spec, params)
    try:
        policy = create_learner(algorithm, spec, params, training=False)
        snapshot = read_checkpoint(checkpoint)
        policy.load_for_evaluation(snapshot)
        started = time.perf_counter()
        started_unix = time.time()
        started_cpu = time.process_time()
        summary = evaluator.evaluate(policy, preserve_rng=False)
        seconds = time.perf_counter() - started
        summary.update(policy_inference_device=str(policy.device),
                       snapshot_sha256=snapshot["_identity"]["sha256"],
                       evaluation_started_unix=started_unix, evaluation_finished_unix=time.time(),
                       evaluation_cpu_seconds=time.process_time() - started_cpu,
                       evaluation_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = result_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump([summary, seconds], stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(result_path)
    finally:
        evaluator.close()


class BackgroundSelectionEvaluator:
    """Own the process and files for one checkpoint evaluation at a time."""

    def __init__(self, algorithm, scenario, scenario_dir, spec, params, workspace, *, device):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        policy_params = {**params, "device": device, "compile_networks": False}
        self._args = (algorithm, scenario, scenario_dir, spec, policy_params)
        self._process = None

    def snapshot(self, version):
        return self.workspace / f"{version:09d}.pt"

    def start(self, version):
        snapshot = self.snapshot(version)
        result = snapshot.with_suffix(".json")
        if result.is_file():
            return
        process = get_context("spawn").Process(target=_evaluate_selection_checkpoint,
                                               args=(*self._args, snapshot, result))
        process.start()
        self._process = process

    def result(self, version, *, wait=False):
        process = self._process
        if process is not None:
            if wait:
                process.join()
            status = process.exitcode
            if status is None:
                return None
            self.close()
            if status:
                raise RuntimeError(f"Evaluation of PPO update {version} exited with status {status}; resume to retry")
        return json.loads(self.snapshot(version).with_suffix(".json").read_text(encoding="utf-8"))

    def prune(self, versions):
        keep = {self.snapshot(version).stem for version in versions}
        for path in self.workspace.iterdir():
            if path.is_file() and (path.stem not in keep or path.suffix == ".tmp"):
                path.unlink()

    def close(self):
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        process.join()
        process.close()
        self._process = None
