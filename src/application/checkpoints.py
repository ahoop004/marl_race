"""Checkpoint path resolution at the application boundary."""
from pathlib import Path


def resolve_scenario_relative_path(value: str, scenario_dir: Path) -> Path:
    """Resolve checkpoint/config paths relative to the declaring scenario."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (scenario_dir / path).resolve()


def resolve_checkpoint_path(value: str) -> Path:
    """CLI paths are relative to cwd; a run directory selects its best model."""
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


