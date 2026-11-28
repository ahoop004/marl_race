"""Checkpoint path resolution at the application boundary."""
from pathlib import Path


def resolve_scenario_relative_path(value: str, scenario_dir: Path) -> Path:
    """Resolve checkpoint/config paths relative to the declaring scenario."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (scenario_dir / path).resolve()


def resolve_checkpoint_path(value: str, *, operation="evaluation", base_dir=None) -> Path:
    """Evaluation prefers best.pt and falls back to final.pt.

    Directory-based transfer requires best.pt. Recovery selects latest.pt.
    Relative paths use base_dir when supplied and the current directory otherwise.
    """
    if operation not in {"evaluation", "transfer", "resume"}:
        raise ValueError(f"Unsupported checkpoint operation: {operation!r}")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if path.is_dir():
        directory = path
        path = directory / ("latest.pt" if operation == "resume" else "best.pt")
        if operation == "evaluation" and not path.is_file():
            final = directory / "final.pt"
            if final.is_file():
                path = final
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def validate_resume_experiment(metadata, scenario):
    """Recovery must retain task, reward, opponent and evaluation protocols."""
    from core.provenance import collect_map_protocols
    from utils.torch_io import validate_checkpoint_compatibility
    from copy import deepcopy
    stored = deepcopy(metadata["configuration"])
    if not stored:
        raise ValueError("Application resume requires a fully resolved experiment configuration")
    # The original schema implied empty kwargs. Add only these unambiguous
    # defaults; learner recovery separately validates the optimization contract.
    if "algorithm" in stored:
        stored["algorithm"].setdefault("optimizer_kwargs", {})
        stored["algorithm"]["params"].setdefault("lr_scheduler_kwargs", {})
        if stored["algorithm"]["name"] == "mappo":
            stored["algorithm"]["params"].setdefault("lr_schedule", "constant")
            stored["algorithm"]["params"].setdefault("learning_rate_end", None)
    if bool(stored.get("continual")) != bool(scenario.get("continual")):
        raise ValueError("Recovery must retain continual-learning mode; use transfer for a new experiment")
    fields = ("algorithm", "network", "environment", "agents", "controllers", "maps", "evaluation", "mappo", "continual")
    expected = {key: deepcopy(scenario[key]) for key in fields if key in scenario}
    saved_evaluation = stored.get("evaluation", {})
    current_evaluation = expected.get("evaluation", {})
    if (isinstance(saved_evaluation, dict) and isinstance(current_evaluation, dict)
            and saved_evaluation.get("final_test") is not None
            and current_evaluation.get("final_test") is not None
            and saved_evaluation["final_test"] != current_evaluation["final_test"]):
        from training.evaluation_config import resolve_evaluation_protocol

        # Legacy composition checkpoints used per-map evaluation implicitly.
        saved_evaluation["final_test"] = resolve_evaluation_protocol(stored, "final")
        current_evaluation["final_test"] = resolve_evaluation_protocol(scenario, "final")
    if scenario.get("continual"):
        # This execution limit may change after a task-boundary pilot; the
        # ordered task protocol and every learning parameter remain immutable.
        stored["continual"].pop("stop_after_task", None)
        expected["continual"].pop("stop_after_task", None)
    validate_checkpoint_compatibility(stored, expected)
    validate_checkpoint_compatibility(stored["experiment"], {"seed": scenario["experiment"].get("seed")})
    if scenario.get("continual"):
        validate_checkpoint_compatibility(stored["experiment"], {
            key: scenario["experiment"].get(key) for key in
            ("total_steps", "parallel_envs", "envs_per_worker", "collection_backend")})
    current_maps = collect_map_protocols(scenario["environment"])
    stored_provenance = metadata["provenance"]
    # Older provenance only recorded YAML. Retain those checks without making
    # additive geometry metadata invalidate otherwise compatible recovery.
    if stored_provenance.get("version") in (None, "1.0"):
        for name, details in stored_provenance.get("map_protocols", {}).items():
            if "assets" not in details and name in current_maps:
                current_maps[name].pop("assets", None)
    validate_checkpoint_compatibility(stored_provenance, {"map_protocols": current_maps})
