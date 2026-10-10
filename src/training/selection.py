"""Build isolated checkpoint selection tasks without disturbing training RNGs."""
from copy import deepcopy

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
