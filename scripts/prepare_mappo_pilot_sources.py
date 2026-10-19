#!/usr/bin/env python3
"""Preserve the two native PPO best.pt sources for the Phase 6 MAPPO pilot.

Run on Wahab's shared ~/marl_race checkout. No Ray startup or training occurs.
An existing destination is verified and reused, never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agents.common.checkpoints import read_checkpoint
from core.scenario import load_and_expand_scenario
from ray_tune_sources import inspect_model, read_sources, source_record, verify_source
from ray_tune_sweeps import ranking, read_spec
from ray_tune_trials import copy_checkpoint, exclusive

PRESETS = ("mlp_base", "cnn1d_base")
SEED = 42


def select_sources(sweep_root):
    """Match trial metadata, then revalidate only those native selection results."""
    summary = json.loads((sweep_root / "ranked_summary.json").read_text())
    selected = {}
    for path in sorted((sweep_root / "trial_runs").glob("*/ray_trial.json")):
        identity = json.loads(path.read_text())
        entry = identity["settings"].get("sweep_trial", {})
        preset = entry.get("network_preset")
        if identity["algorithm"] != "ppo" or preset not in PRESETS or entry.get("seed") != SEED:
            continue
        if preset in selected:
            raise ValueError(f"Ambiguous PPO seed-{SEED} trials for {preset}; select explicitly before preparing sources")
        matches = [row for row in summary["ranked_trials"]
                   if Path(row["training_output_dir"]).resolve() == path.parent.resolve()]
        if len(matches) != 1:
            raise ValueError(f"Trial missing or ambiguous in native ranked results: {path.parent}")
        checked = ranking(matches)
        if len(checked["ranked_trials"]) != 1 or checked["diagnostics"]:
            raise ValueError(f"Invalid native PPO selection: {checked['diagnostics']}")
        best = Path(checked["ranked_trials"][0]["best_checkpoint"]).resolve()
        metadata = read_checkpoint(best)["metadata"]
        config = metadata["configuration"]
        if (config["wandb"]["tune"].get("network_preset") != preset
                or config["experiment"]["seed"] != SEED or config["environment"]["lidar_beams"] != 1080
                or entry["parameters"].get("network") != preset):
            raise ValueError(f"PPO source preset/seed/1080-beam metadata differs: {best}")
        selected[preset] = (best, source_record(best, preset, SEED))
    if set(selected) != set(PRESETS):
        raise ValueError(f"Missing ranked PPO seed-{SEED} trials: {sorted(set(PRESETS) - set(selected))}")
    return selected


def prepare(sweep_root, destination):
    if not sweep_root.is_dir():
        raise ValueError(f"Completed PPO sweep unavailable: {sweep_root}. Run this command on Wahab with the original shared outputs.")
    selected = select_sources(sweep_root)
    pilot = read_spec(REPO / "scripts/sweeps/mappo_pilot_phase6.yaml", REPO)
    # Reuse native complete-source/transfer checks before publishing copies.
    for preset, (path, record) in selected.items():
        for arm in ("full_finetune", "lora"):
            config = load_and_expand_scenario(pilot["scenario"], overrides=[*pilot["fixed_overrides"],
                f"network={preset}", f"adaptation={arm}", f"adaptation.checkpoint={json.dumps(str(path))}"])
            inspect_model(config, record)
    records = [{**selected[preset][1], "checkpoint": str(destination / f"{preset}_ppo_seed{SEED}_best.pt")}
               for preset in PRESETS]
    manifest = destination / "manifest.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(destination.parent / f".{destination.name}.lock"):
        if destination.exists():
            expected = {(r["network"], r["pretraining_seed"]): r for r in records}
            if read_sources(manifest) != expected:
                raise ValueError(f"Existing source directory differs; refusing to overwrite: {destination}")
        else:
            staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
            try:
                for record in records:
                    path = staging / Path(record["checkpoint"]).name
                    copy_checkpoint(selected[record["network"]][0], path)
                    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                        raise ValueError(f"Source changed while copying: {record['network']}")
                    path.chmod(0o444)
                temporary_manifest = staging / "manifest.yaml"
                temporary_manifest.write_text(yaml.safe_dump({"schema_version": 1, "sources": records}, sort_keys=False))
                temporary_manifest.chmod(0o444)
                staging.rename(destination)
                destination.chmod(0o555)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        for record in records:
            verify_source(record)
    for record in records:
        print(f"{record['network']} PPO seed {SEED}\n  selected: {selected[record['network']][0]}"
              f"\n  preserved: {record['checkpoint']}\n  SHA-256: {record['sha256']}")
    print(f"Verified source manifest: {manifest}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=REPO / "outputs/ray_tune_phase6-ppo-q1wang-4gpu-v1")
    parser.add_argument("--destination", type=Path, default=REPO / "outputs/pretrained_sources/phase6-mappo-pilot-v1")
    args = parser.parse_args()
    try:
        prepare(args.sweep_root.expanduser().resolve(), args.destination.expanduser().resolve())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Source preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
