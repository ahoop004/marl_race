"""Selection and final-test protocols for racing experiments."""
from typing import Any, Dict

from core.scenario import ScenarioError


EVALUATION_STRATEGIES = frozenset({"completion_progress", "lap_time", "team_completion"})


def resolve_evaluation_protocol(scenario: Dict[str, Any], protocol: str) -> Dict[str, Any]:
    """Resolve fixed selection/final seeds without mutating training config."""
    evaluation = scenario.get("evaluation", {}) or {}
    if not isinstance(evaluation, dict):
        raise ScenarioError("'evaluation' must be a dictionary.")
    if evaluation.get("selection_strategy", "completion_progress") not in EVALUATION_STRATEGIES:
        raise ScenarioError("evaluation.selection_strategy must be completion_progress, lap_time or team_completion")
    if "progress_agent_id" in evaluation:
        raise ScenarioError("evaluation.progress_agent_id is unsupported for completion experiments")
    for key in ("terminate_on_track_limit", "terminate_on_collision", "lap_completion"):
        if key in evaluation and not isinstance(evaluation[key], bool):
            raise ScenarioError(f"evaluation.{key} must be boolean")
    for key in ("target_laps", "every_steps", "every_episodes"):
        value = evaluation.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ScenarioError(f"evaluation.{key} must be a positive integer")
    if protocol not in {"selection", "final"}:
        raise ScenarioError(f"Unknown evaluation protocol: {protocol!r}.")
    selection = {
        "seed": evaluation.get("seed", int(scenario["experiment"].get("seed", 0) or 0) + 10_000),
        "episodes": evaluation.get("episodes", 8),
    }
    final = evaluation.get("final_test")
    if final is not None and (not isinstance(final, dict) or not {"seed", "episodes"} <= final.keys()):
        raise ScenarioError("'evaluation.final_test' requires explicit seed and episodes.")
    if final is not None and set(final) - {"seed", "episodes", "target_laps", "max_steps"}:
        raise ScenarioError("'evaluation.final_test' accepts seed, episodes, target_laps and max_steps.")
    for key in ("target_laps", "max_steps"):
        value = (final or {}).get(key)
        minimum = 0 if key == "max_steps" else 1
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
            raise ScenarioError(f"evaluation.final_test.{key} must be an integer >= {minimum}")
    for name, config in (("selection", selection), ("final", final)):
        if config is None:
            continue
        for key, minimum in (("seed", 0), ("episodes", 1)):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ScenarioError(f"Evaluation {name} {key} must be an integer >= {minimum}.")
        if config["seed"] + config["episodes"] > 2**32:
            raise ScenarioError(f"Evaluation {name} seeds exceed the NumPy seed range.")
    if final is not None and max(selection["seed"], final["seed"]) < min(
        selection["seed"] + selection["episodes"], final["seed"] + final["episodes"]
    ):
        raise ScenarioError("Checkpoint-selection and final-test seed ranges must be disjoint.")
    if protocol == "final" and final is None:
        raise ScenarioError("--eval-protocol final requires evaluation.final_test.")
    config = selection if protocol == "selection" else final
    # Final evaluation may override the horizon; otherwise inherit selection.
    max_steps = evaluation.get("max_steps")
    if max_steps is None:
        max_steps = scenario["environment"].get("max_steps", 5000)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ScenarioError("Evaluation max_steps must be a nonnegative integer.")
    result = {"name": protocol, "seed": config["seed"], "episodes": config["episodes"],
              "max_steps": config.get("max_steps", max_steps)}
    target_laps = config.get("target_laps")
    if target_laps is not None:
        result["target_laps"] = target_laps
    return result


