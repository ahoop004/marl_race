"""Build isolated checkpoint selection tasks without disturbing training RNGs."""
from copy import deepcopy
import json
from multiprocessing import get_context
from pathlib import Path
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
    from training.algorithms import create_learner

    torch.set_num_threads(1)
    evaluator = create_selection_evaluator(algorithm, scenario, scenario_dir, spec, params)
    try:
        policy = create_learner(algorithm, spec, params, training=False)
        policy.load_for_evaluation(checkpoint)
        started = time.perf_counter()
        summary = evaluator.evaluate(policy, preserve_rng=False)
        summary["policy_inference_device"] = str(policy.device)
        result_path.write_text(json.dumps([summary, time.perf_counter() - started]), encoding="utf-8")
    finally:
        evaluator.close()


class BackgroundSelectionEvaluator:
    """Own the process and files for one checkpoint evaluation at a time."""

    def __init__(self, algorithm, scenario, scenario_dir, spec, params, workspace, *, device):
        self.snapshot = Path(workspace) / "candidate.pt"
        self._result_path = Path(workspace) / "result.json"
        policy_params = {**params, "device": device, "compile_networks": False}
        self._args = (algorithm, scenario, scenario_dir, spec, policy_params, self.snapshot, self._result_path)
        self._process = None

    def start(self):
        process = get_context("spawn").Process(target=_evaluate_selection_checkpoint, args=self._args)
        process.start()
        self._process = process

    def result(self, *, wait=False):
        process = self._process
        if wait:
            process.join()
        if process.exitcode is None:
            return None
        status = process.exitcode
        self.close()
        if status:
            raise RuntimeError(f"Evaluation process exited with status {status}")
        return json.loads(self._result_path.read_text(encoding="utf-8"))

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
