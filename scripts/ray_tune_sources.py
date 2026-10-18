"""Immutable PPO source records shared by sweep preflight and remote trial starts.

Print a manifest from a real, natively selected PPO checkpoint with:
  PYTHONPATH=src python3 scripts/ray_tune_sources.py CHECKPOINT --network mlp_base --pretraining-seed 42
Paths in manifests are relative to the manifest file; exported paths are absolute.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import yaml

NETWORKS = {"mlp_small", "mlp", "mlp_base", "mlp_wide", "mlp_deep", "resmlp_base", "cnn1d_base", "rescnn1d_base"}
ADAPTATIONS = {"scratch", "full_finetune", "lora"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def configuration_digest(config):
    value = deepcopy(config)
    value.pop("_configuration", None)
    value.pop("wandb", None)  # Run IDs and attempt/node metadata are assigned later.
    value.get("checkpoint", {}).pop("resume", None)
    return fingerprint(value)


def source_record(path, network, pretraining_seed):
    """Derive identity from bytes and native PPO selection, never MAPPO results."""
    from agents.common.checkpoints import read_checkpoint
    if not isinstance(network, str) or network not in NETWORKS:
        raise ValueError(f"Unknown network preset: {network}")
    if type(pretraining_seed) is not int or not 0 <= pretraining_seed < 2**32:
        raise ValueError("pretraining_seed must be an integer in the NumPy seed range")
    payload = read_checkpoint(path)
    meta = payload["metadata"]
    if meta["algorithm"] != "ppo" or meta["adaptation"]["lora"] is not None:
        raise ValueError("Source must be a complete PPO checkpoint")
    config = meta["configuration"]
    if config.get("experiment", {}).get("seed") != pretraining_seed:
        raise ValueError("Source pretraining seed differs from checkpoint configuration")
    selected = meta["selection"]
    if (not isinstance(selected, dict) or selected.get("selection_strategy") != "completion_progress"
            or selected.get("evaluation_protocol", {}).get("name") != "selection" or not selected.get("is_best")):
        raise ValueError("Source requires a native PPO selection identity (use selected best.pt, never final-test results)")
    recorded = config.get("wandb", {}).get("tune", {}).get("network_preset")
    if recorded is not None and recorded != network:
        raise ValueError("Source network preset differs from checkpoint provenance")
    return {"network": network, "pretraining_seed": pretraining_seed,
            "checkpoint": payload["_identity"]["path"], "sha256": payload["_identity"]["sha256"],
            "source_git": (meta["provenance"].get("git") or {}).get("commit"),
            "architecture_sha256": fingerprint(meta["network"]), "selection_sha256": fingerprint(selected)}


def read_sources(path):
    manifest = Path(path)
    if not manifest.is_file():
        raise ValueError(f"Missing PPO source manifest: {manifest}; populate scripts/sweeps/ppo_sources.template.yaml")
    value = yaml.safe_load(manifest.read_text())
    if (not isinstance(value, dict) or set(value) != {"schema_version", "sources"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise ValueError("Source manifest requires schema_version: 1 and sources: [records]")
    if not isinstance(value["sources"], list):
        raise ValueError("Source manifest sources must be a list")
    records = {}
    fields = {"network", "pretraining_seed", "checkpoint", "sha256", "source_git", "architecture_sha256", "selection_sha256"}
    for record in value["sources"]:
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError(f"Source record requires {sorted(fields)}; generate records with scripts/ray_tune_sources.py")
        key = (record["network"], record["pretraining_seed"])
        if (not isinstance(record["network"], str) or record["network"] not in NETWORKS or type(record["pretraining_seed"]) is not int
                or not 0 <= record["pretraining_seed"] < 2**32 or key in records):
            raise ValueError(f"Invalid or duplicate source key: {key}")
        for field in ("sha256", "architecture_sha256", "selection_sha256"):
            if not isinstance(record[field], str) or not re.fullmatch("[0-9a-f]{64}", record[field]):
                raise ValueError(f"Source {key} needs a real {field}")
        if not isinstance(record["checkpoint"], str) or not record["checkpoint"].strip():
            raise ValueError(f"Source {key} needs an existing checkpoint path")
        if record["source_git"] is not None and not isinstance(record["source_git"], str):
            raise ValueError(f"Source {key} source_git must be a Git revision or null")
        resolved = Path(record["checkpoint"]).expanduser()
        record = {**record, "checkpoint": str((manifest.parent / resolved).resolve())}
        records[key] = record
    return records


def verify_source(record):
    path = Path(record["checkpoint"])
    if not path.is_file():
        raise ValueError(f"Missing PPO source: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"PPO source SHA-256 changed: {path}")
    actual = source_record(path, record["network"], record["pretraining_seed"])
    if actual != record:
        raise ValueError(f"PPO source metadata/selection differs from manifest: {path}")


def inspect_model(config, source=None):
    """Build CPU inference models using the real task specs; no reset or collector.

    Native transfer validates the complete PPO actor AND critic, physics, actions,
    driving observations, and explicit neighbor extension before loading weights.
    """
    import torch
    from core.task_builder import create_race_task
    from training.algorithms import learner_params, create_learner
    from agents.common.networks import model_spec
    from core.provenance import learner_parameter_counts

    algorithm = config["algorithm"]["name"]
    task = create_race_task(config)
    try:
        params = learner_params(config, task.spec, algorithm)
        params.update(device="cpu", compile_networks=False)
        with torch.random.fork_rng(devices=[]):
            learner = create_learner(algorithm, task.spec, params, training=False)
            if source:
                verify_source(source)
                if config["adaptation"]["observation_extension"] != "frenet_neighbors":
                    raise ValueError("MAPPO sweep transfer requires explicit frenet_neighbors extension")
                learner.initialize_from_checkpoint(source["checkpoint"], scope=config["adaptation"]["load_scope"],
                    observation_extension=config["adaptation"]["observation_extension"])
                if learner.source_checkpoint["sha256"] != source["sha256"]:
                    raise ValueError("Loaded PPO source SHA-256 changed during validation")
            counts = learner_parameter_counts(learner)
            return {"resolved_network": learner.network_config,
                    "encoder_type": model_spec(learner.network_config, "actor")["encoder"]["type"],
                    "actor_head": model_spec(learner.network_config, "actor")["architecture"],
                    "critic_head": model_spec(learner.network_config, "critic")["architecture"],
                    "observation_dims": task.spec.observation_dims, "lidar_beams": config["environment"]["lidar_beams"],
                    "actor_sharing": config["network"]["actor"]["parameter_sharing"],
                    "lora": config["adaptation"]["lora"], "parameter_counts": counts}
    finally:
        task.close()


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--network", required=True, choices=sorted(NETWORKS))
    parser.add_argument("--pretraining-seed", required=True, type=int)
    args = parser.parse_args()
    print(yaml.safe_dump({"schema_version": 1, "sources": [source_record(args.checkpoint, args.network, args.pretraining_seed)]}, sort_keys=False))
