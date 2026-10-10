"""Checkpoint path resolution at the application boundary."""
from pathlib import Path


def resolve_scenario_relative_path(value: str, scenario_dir: Path) -> Path:
    """Resolve checkpoint/config paths relative to the declaring scenario."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (scenario_dir / path).resolve()


def resolve_checkpoint_path(value: str, *, operation="evaluation", base_dir=None) -> Path:
    """Directories select best.pt, or latest.pt for recovery.

    CLI callers pass canonical paths. Configured relative paths may name their
    declaring scenario directory; standalone callers retain cwd semantics.
    """
    if operation not in {"evaluation", "transfer", "resume"}:
        raise ValueError(f"Unsupported checkpoint operation: {operation!r}")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if path.is_dir():
        path = path / ("latest.pt" if operation == "resume" else "best.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def validate_resume_experiment(metadata, scenario):
    """Recovery must retain task, reward, opponent and evaluation protocols."""
    from core.provenance import collect_map_protocols
    from utils.torch_io import validate_checkpoint_compatibility
    stored = metadata["configuration"]
    if not stored:
        raise ValueError("Application resume requires a fully resolved experiment configuration")
    fields = ("algorithm", "network", "environment", "agents", "controllers", "maps", "evaluation", "mappo")
    validate_checkpoint_compatibility(stored, {key: scenario[key] for key in fields if key in scenario})
    validate_checkpoint_compatibility(stored["experiment"], {"seed": scenario["experiment"].get("seed")})
    current_maps = collect_map_protocols(scenario["environment"])
    stored_provenance = metadata["provenance"]
    # Older provenance only recorded YAML. Retain those checks without making
    # additive geometry metadata invalidate otherwise compatible recovery.
    if stored_provenance.get("version") in (None, "1.0"):
        for name, details in stored_provenance.get("map_protocols", {}).items():
            if "assets" not in details and name in current_maps:
                current_maps[name].pop("assets", None)
    validate_checkpoint_compatibility(stored_provenance, {"map_protocols": current_maps})
