"""Versioned model checkpoints and explicit evaluation, transfer and recovery."""
from pathlib import Path
import hashlib
import random

import numpy as np
import torch
import torchrl

from utils.torch_io import atomic_save, safe_load, validate_checkpoint_compatibility

SCHEMA_VERSION = 1
FORMAT = "marl_race"
RESUME_BOUNDARY = "completed_update_fresh_episode"


def _plain(value):
    """Keep the payload within weights_only's tensors and primitive containers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.tensor(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def policy_metadata(policy):
    algorithm = policy.algorithm
    ids = list(policy.agent_ids)
    observations = (policy.observation_contracts if algorithm == "mappo"
                    else dict.fromkeys(ids, policy.observation_contract))
    dims = policy.obs_dims if algorithm == "mappo" else dict.fromkeys(ids, policy.obs_dim)
    return {
        "algorithm": algorithm,
        "implementation": policy.implementation,
        "network": dict(policy.network_config),
        "structure": {
            "encoder": "identity", "distribution": "squashed_gaussian_pre_tanh_v1",
            "actor_mode": getattr(policy, "actor_mode", "independent"),
            "critic_mode": getattr(policy, "critic_mode", "local_observation"),
            "agent_ids": ids, "groups": {"agents": ids},
            "routing": {aid: (i if getattr(policy, "actor_mode", "independent") == "independent" else 0)
                        for i, aid in enumerate(ids)},
        },
        "contracts": {
            "observations": {aid: {"dim": dims[aid], "contract": observations[aid]} for aid in ids},
            "actions": {"dim": policy.action_dim, "low": policy.action_low, "high": policy.action_high,
                        "contract": policy.action_contract},
            "physics": policy.physics_contract,
            "global_state": ({"dim": policy.global_state_dim, "version": policy.global_state_contract_version}
                             if algorithm == "mappo" else None),
        },
        "adaptation": {
            "method": ("lora" if getattr(policy, "lora_config", None) else
                       "full_finetune" if policy.source_checkpoint else "scratch"),
            "lora": getattr(policy, "lora_contract", None), "source_checkpoint": policy.source_checkpoint,
        },
        "versions": {"torch": str(torch.__version__), "torchrl": str(torchrl.__version__)},
    }


def save_checkpoint(policy, path, *, provenance=None, configuration=None, training_state=None,
                    progress=None, selection=None):
    require_source = getattr(policy, "_require_lora_source", None)
    if require_source:
        require_source()
    metadata = policy_metadata(policy)
    metadata.update(provenance=provenance or {}, configuration=configuration or {},
                    progress=progress or {}, selection=selection)
    payload = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "metadata": metadata,
               "models": {"actor": policy.actor.state_dict(), "critic": policy.critic.state_dict()},
               "training": training_state}
    atomic_save(_plain(payload), path)


def read_checkpoint(path):
    with Path(path).open("rb") as handle:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        handle.seek(0)
        checkpoint = safe_load(handle, map_location="cpu")
    _validate_schema(checkpoint)
    checkpoint["_identity"] = {"path": str(Path(path).resolve()), "sha256": digest}
    return checkpoint


def _validate_schema(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint format; historical checkpoints must be retrained")
    if type(checkpoint.get("schema_version")) is not int or checkpoint["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema version: {checkpoint.get('schema_version')!r}")
    for section in ("metadata", "models"):
        if not isinstance(checkpoint.get(section), dict):
            raise ValueError(f"Checkpoint missing {section} section")
    required = {"algorithm", "implementation", "network", "structure", "contracts", "adaptation",
                "versions", "provenance", "configuration", "progress", "selection"}
    if not required <= checkpoint["metadata"].keys() or not {"actor", "critic"} <= checkpoint["models"].keys():
        raise ValueError("Checkpoint missing required model metadata or state")
    if "training" not in checkpoint:
        raise ValueError("Checkpoint missing training section (use null for model-only checkpoints)")


def _payload(source):
    if isinstance(source, (str, Path)):
        return read_checkpoint(source)
    _validate_schema(source)
    return source


def _validate_models(policy, checkpoint, *, critic=False):
    expected = _plain(policy_metadata(policy))
    stored = checkpoint["metadata"]
    validate_checkpoint_compatibility(stored, {key: expected[key] for key in (
        "algorithm", "implementation", "network", "structure", "contracts")})
    validate_checkpoint_compatibility(stored["adaptation"], {"lora": expected["adaptation"]["lora"]})
    states = [("actor", checkpoint["models"]["actor"], policy.actor.state_dict())]
    if critic:
        states.append(("critic", checkpoint["models"]["critic"], policy.critic.state_dict()))
    validate_checkpoint_compatibility(checkpoint, {}, states=states)


def load_for_evaluation(policy, source):
    checkpoint = _payload(source)
    _validate_models(policy, checkpoint)
    policy.actor.load_state_dict(checkpoint["models"]["actor"], strict=True)
    policy.source_checkpoint = checkpoint["metadata"]["adaptation"]["source_checkpoint"]
    if hasattr(policy, "_lora_ready"):
        policy._lora_ready = True
    policy.actor.eval()
    return checkpoint["metadata"]


def evaluation_routing(checkpoint):
    """Reconstruct saved actor ownership; task/model contracts still validate."""
    metadata = checkpoint["metadata"]
    lora = metadata["adaptation"]["lora"]
    return {"actor_mode": metadata["structure"]["actor_mode"],
            "lora": ({key: value for key, value in lora.items() if key in {
                "mode", "rank", "alpha", "train_log_std", "per_agent_log_std"}} if lora else None)}


def _observation_transfer(source, destination, extension):
    if source == destination:
        return False
    # Contracts may contain tensors in future; use the common comparator.
    if source["dim"] == destination["dim"]:
        validate_checkpoint_compatibility(source, destination)
        return False
    if extension != "frenet_neighbors" or source["dim"] >= destination["dim"]:
        raise ValueError("Incompatible observation dimensions; only explicit frenet_neighbors expansion is supported")
    from agents.common.observations import observation_layout
    prefix, driving, total = observation_layout(source["contract"])
    dest_prefix, dest_driving, dest_total = observation_layout(destination["contract"])
    if (source["dim"] != driving or total != driving or dest_driving != driving
            or dest_total != destination["dim"]
            or not destination["contract"]["observation"].get("frenet_neighbors", {}).get("enabled")):
        raise ValueError("Incompatible Frenet observation prefix or dimensions")
    validate_checkpoint_compatibility({"observation_prefix": prefix}, {"observation_prefix": dest_prefix})
    return True


def initialize_from_checkpoint(policy, path, *, scope, observation_extension=None):
    """Transfer into a newly constructed policy, retaining fresh training state."""
    if scope not in {"actor_only", "actor_and_critic"}:
        raise ValueError("Transfer scope must be actor_only or actor_and_critic")
    if policy.optimizer is not None and policy.optimizer.state:
        raise ValueError("Transfer requires a fresh learner; use resume_training to continue an existing experiment")
    checkpoint = read_checkpoint(path)
    stored, expected = checkpoint["metadata"], _plain(policy_metadata(policy))
    if stored["algorithm"] != "ppo" or stored["adaptation"]["lora"] is not None:
        raise ValueError("Transfer requires a complete PPO source policy")
    from agents.common.ppo_policy import PPOPolicy
    validate_checkpoint_compatibility(stored, {"implementation": PPOPolicy.implementation})
    if scope == "actor_and_critic" and expected["algorithm"] != "ppo":
        raise ValueError("MAPPO transfer supports actor_only; its centralized critic must be fresh")
    validate_checkpoint_compatibility(stored["network"], {key: expected["network"][key] for key in (
        "architecture", "actor_hidden_dims", "activation")})
    validate_checkpoint_compatibility(stored["structure"], {key: expected["structure"][key]
                                                          for key in ("encoder", "distribution")})
    validate_checkpoint_compatibility(stored["contracts"], {
        "actions": expected["contracts"]["actions"], "physics": expected["contracts"]["physics"]})
    observations = stored["contracts"]["observations"]
    if len(observations) != 1:
        raise ValueError("PPO transfer requires one source observation contract")
    source = next(iter(observations.values()))
    expansions = [_observation_transfer(source, target, observation_extension)
                  for target in expected["contracts"]["observations"].values()]
    recipient = policy.actor.actors[policy.agent_ids[0]] if getattr(policy, "actor_mode", None) == "independent" and expected["algorithm"] == "mappo" else policy.actor
    actor_expected = recipient.base_state_dict() if getattr(policy, "lora_config", None) else recipient.state_dict()
    from agents.common.networks import expand_observation_inputs
    actor_state = expand_observation_inputs(checkpoint["models"]["actor"], actor_expected,
        policy.network_config, source_width=source["dim"], expand_inputs=any(expansions))
    states = [("actor", actor_state, actor_expected)]
    critic_state = None
    if scope == "actor_and_critic":
        validate_checkpoint_compatibility(stored["network"], expected["network"])
        critic_state = expand_observation_inputs(checkpoint["models"]["critic"], policy.critic.state_dict(),
            policy.network_config, source_width=source["dim"], expand_inputs=any(expansions))
        states.append(("critic", critic_state, policy.critic.state_dict()))
    validate_checkpoint_compatibility(checkpoint, {}, states=states)
    if getattr(policy, "lora_config", None):
        policy.actor.reset_adapters()
        if policy.lora_config.get("per_agent_log_std"):
            actor_state.update({f"log_stds.{i}": actor_state["log_std"].clone() for i in range(len(policy.agent_ids))})
        policy.actor.load_state_dict({**policy.actor.state_dict(), **actor_state}, strict=True)
        policy._lora_ready = True
    elif expected["algorithm"] == "mappo" and policy.actor_mode == "independent":
        for actor in policy.actor.actors.values():
            actor.load_state_dict(actor_state, strict=True)
    else:
        policy.actor.load_state_dict(actor_state, strict=True)
    if critic_state is not None:
        policy.critic.load_state_dict(critic_state, strict=True)
    policy.set_training_progress(0)
    policy.source_checkpoint = {**checkpoint["_identity"], "algorithm": "ppo", "scope": scope}


def capture_rng():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"generator": numpy[0],
            "keys": torch.tensor(numpy[1].astype(np.int64)), "position": numpy[2],
            "has_gauss": numpy[3], "cached_gaussian": numpy[4]},
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    numpy = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((numpy["generator"], numpy["keys"].numpy().astype(np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.random.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the checkpoint's CUDA RNG device topology")
        torch.cuda.set_rng_state_all(state["cuda"])


def optimization_contract(policy):
    names = ("lr", "lr_end", "lr_schedule", "target_kl", "gamma", "gae_lambda", "clip_range",
             "ent_coef", "vf_coef", "max_grad_norm", "n_steps", "n_epochs", "batch_size")
    return {**{key: getattr(policy, key) for key in names},
            "min_rollout_steps": getattr(policy, "min_rollout_steps", 1),
            **({key: getattr(policy, key) for key in ("reward_mode", "team_return_mode", "team_reward_reduction")}
               if hasattr(policy, "reward_mode") else {})}


def resume_training(policy, path):
    checkpoint = _payload(path)
    state = checkpoint["training"]
    if not isinstance(state, dict) or state.get("boundary") != RESUME_BOUNDARY:
        raise ValueError("Checkpoint is model-only or has an unsupported resume boundary; use latest.pt")
    _validate_models(policy, checkpoint, critic=True)
    validate_checkpoint_compatibility(state, {"optimization_contract": optimization_contract(policy)})
    validate_checkpoint_compatibility(checkpoint["metadata"]["versions"], policy_metadata(policy)["versions"])
    if policy.optimizer is None:
        raise ValueError("Training resume requires a training learner")
    required = {"optimizer", "schedule_progress", "progress", "budget", "rng", "hooks", "recovery_seed"}
    if not required <= state.keys():
        raise ValueError("Checkpoint missing required recovery state")
    policy.actor.load_state_dict(checkpoint["models"]["actor"], strict=True)
    policy.critic.load_state_dict(checkpoint["models"]["critic"], strict=True)
    policy.optimizer.load_state_dict(state["optimizer"])
    policy.set_training_progress(state["schedule_progress"])
    policy.source_checkpoint = checkpoint["metadata"]["adaptation"]["source_checkpoint"]
    if hasattr(policy, "_lora_ready"):
        policy._lora_ready = True
    # Restore RNG after task/collector construction, which may consume draws.
    return state, checkpoint["metadata"]
