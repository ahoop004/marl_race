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
    from agents.common.networks import model_spec
    from agents.common.observations import observation_segments

    algorithm = policy.algorithm
    ids = list(policy.agent_ids)
    observations = (policy.observation_contracts if algorithm == "mappo"
                    else dict.fromkeys(ids, policy.observation_contract))
    dims = policy.obs_dims if algorithm == "mappo" else dict.fromkeys(ids, policy.obs_dim)
    encoders = {role: model_spec(policy.network_config, role)["encoder"]["type"]
                for role in ("actor", "critic")}
    convolutional = any(kind != "identity" for kind in encoders.values())
    segments = ({"actor": ({aid: observation_segments(observations[aid], dims[aid]) for aid in ids}
                           if encoders["actor"] != "identity" else None),
                 "critic": (observation_segments(policy.observation_contract, policy.obs_dim)
                            if encoders["critic"] != "identity" else None)} if convolutional else {})
    return {
        "algorithm": algorithm,
        "implementation": policy.implementation,
        "network": dict(policy.network_config),
        "structure": {
            "encoder": encoders if convolutional else "identity",
            **({"observation_segments": segments} if convolutional else {}),
            "distribution": "squashed_gaussian_pre_tanh_v1",
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
    if extension not in (None, "frenet_neighbors"):
        raise ValueError("Unsupported observation extension; only frenet_neighbors is supported")
    from agents.common.observations import observation_layout, observation_segments
    for observation in (source, destination):
        if (not isinstance(observation, dict) or set(observation) != {"dim", "contract"}
                or type(observation["dim"]) is not int or observation["dim"] <= 0):
            raise ValueError("Transfer requires explicit observation dimensions and contracts")
        # Historical, unextended MLP checkpoints can have a null contract.
        if observation["contract"] is not None:
            observation_segments(observation["contract"], observation["dim"])
    if source["dim"] == destination["dim"]:
        validate_checkpoint_compatibility(source, destination)
        return False
    if extension != "frenet_neighbors" or source["dim"] >= destination["dim"]:
        raise ValueError("Incompatible observation dimensions; only explicit frenet_neighbors expansion is supported")
    prefix, driving, total = observation_layout(source["contract"])
    dest_prefix, dest_driving, dest_total = observation_layout(destination["contract"])
    if (source["dim"] != driving or total != driving or dest_driving != driving
            or dest_total != destination["dim"]
            or not destination["contract"]["observation"].get("frenet_neighbors", {}).get("enabled")):
        raise ValueError("Incompatible Frenet observation prefix or dimensions")
    validate_checkpoint_compatibility({"observation_prefix": prefix}, {"observation_prefix": dest_prefix})
    return True


def _validate_ppo_transfer_source(checkpoint, network):
    """Validate a complete source against its own factories, independent of the recipient critic.

    Meta-device templates validate every source tensor without allocating model
    storage or consuming the destination process's initialization RNG stream.
    """
    from agents.common.networks import build_actor, build_critic, model_spec
    from agents.common.observations import observation_segments

    stored = checkpoint["metadata"]
    observations = stored["contracts"].get("observations")
    if not isinstance(observations, dict) or len(observations) != 1:
        raise ValueError("PPO transfer requires one source observation contract")
    aid, observation = next(iter(observations.items()))
    if (not isinstance(observation, dict) or set(observation) != {"dim", "contract"}
            or type(observation["dim"]) is not int or observation["dim"] <= 0):
        raise ValueError("Transfer requires explicit source observation dimensions and contracts")
    encoders = {role: model_spec(network, role)["encoder"]["type"] for role in ("actor", "critic")}
    structure = {"encoder": encoders if any(kind != "identity" for kind in encoders.values()) else "identity",
                 "actor_mode": "independent", "critic_mode": "local_observation",
                 "agent_ids": [aid], "groups": {"agents": [aid]}, "routing": {aid: 0}}
    if structure["encoder"] != "identity":
        segments = observation_segments(observation["contract"], observation["dim"])
        structure["observation_segments"] = {
            "actor": {aid: segments} if encoders["actor"] != "identity" else None,
            "critic": segments if encoders["critic"] != "identity" else None}
    validate_checkpoint_compatibility(stored["structure"], structure)
    with torch.device("meta"):
        actor = build_actor(observation["dim"], stored["contracts"]["actions"]["dim"], network,
                            observation_contract=observation["contract"])
        critic = build_critic(observation["dim"], network, observation_contract=observation["contract"])
    validate_checkpoint_compatibility(checkpoint, {}, states=[
        (role, checkpoint["models"][role], model.state_dict()) for role, model in (("actor", actor), ("critic", critic))])
    return observation


def _prepare_model_transfer(state, expected, source_network, target_network, source_observation,
                            expand_inputs, *, role):
    """Prepare an entire model state, copying a validated driving feature prefix."""
    from agents.common.networks import model_spec, expand_observation_inputs
    from agents.common.encoders import fusion_feature_segments
    from agents.common.observations import observation_segments

    source_spec, target_spec = (model_spec(config, role) for config in (source_network, target_network))
    cnn = source_spec["encoder"]["type"] != "identity"
    # Only the newly introduced neighbor branch may have independent settings.
    specs = [dict(source_spec), dict(target_spec)]
    if cnn and expand_inputs:
        for spec in specs:
            spec["encoder"] = {key: value for key, value in spec["encoder"].items()
                               if key not in {"neighbor_hidden_dims", "neighbor_features"}}
    validate_checkpoint_compatibility({f"{role}_spec": specs[0]}, {f"{role}_spec": specs[1]})
    width = source_observation["dim"]
    if cnn:
        segments = observation_segments(source_observation["contract"], width)
        width = fusion_feature_segments(source_spec["encoder"], segments)["neighbors"][1]
    return expand_observation_inputs(state, expected, target_network, role=role, source_width=width,
        expand_inputs=expand_inputs, new_parameter_prefixes=("encoder.neighbors.",) if cnn and expand_inputs else ())


def initialize_from_checkpoint(policy, path, *, scope, observation_extension=None):
    """Validate and prepare all model states before applying a fresh transfer."""
    if scope not in {"actor_only", "actor_and_critic"}:
        raise ValueError("Transfer scope must be actor_only or actor_and_critic")
    if policy.optimizer is not None and policy.optimizer.state:
        raise ValueError("Transfer requires a fresh learner; use resume_training to continue an existing experiment")
    checkpoint = read_checkpoint(path)
    stored, expected = checkpoint["metadata"], _plain(policy_metadata(policy))
    if stored["algorithm"] != "ppo" or stored["adaptation"]["lora"] is not None:
        raise ValueError("Transfer requires a complete PPO source policy")
    from agents.common.ppo_policy import PPOPolicy
    from agents.common.networks import resolve_network_config
    validate_checkpoint_compatibility(stored, {"implementation": PPOPolicy.implementation})
    if scope == "actor_and_critic" and expected["algorithm"] != "ppo":
        raise ValueError("MAPPO transfer supports actor_only; its centralized critic must be fresh")
    source_network = resolve_network_config({"network": stored["network"]}, default_hidden_dims=[])
    validate_checkpoint_compatibility(stored, {"network": source_network})
    source = _validate_ppo_transfer_source(checkpoint, source_network)
    validate_checkpoint_compatibility(stored["structure"], {"distribution": expected["structure"]["distribution"]})
    validate_checkpoint_compatibility(stored["contracts"], {
        "actions": expected["contracts"]["actions"], "physics": expected["contracts"]["physics"]})
    expansions = [_observation_transfer(source, target, observation_extension)
                  for target in expected["contracts"]["observations"].values()]
    independent = expected["algorithm"] == "mappo" and policy.actor_mode == "independent"
    recipients = list(policy.actor.actors.values()) if independent else [policy.actor]
    prepared, states = [], []
    for index, actor in enumerate(recipients):
        target = actor.base_state_dict() if getattr(policy, "lora_config", None) else actor.state_dict()
        state = _prepare_model_transfer(checkpoint["models"]["actor"], target, source_network,
            policy.network_config, source, any(expansions), role="actor")
        prepared.append((actor, state))
        states.append((f"actor.{index}", state, target))
    critic_state = None
    if scope == "actor_and_critic":
        critic_state = _prepare_model_transfer(checkpoint["models"]["critic"], policy.critic.state_dict(),
            source_network, policy.network_config, source, any(expansions), role="critic")
        states.append(("critic", critic_state, policy.critic.state_dict()))
    validate_checkpoint_compatibility(checkpoint, {}, states=states)
    if getattr(policy, "lora_config", None):
        actor_state = prepared[0][1]
        policy.actor.reset_adapters()
        if policy.lora_config.get("per_agent_log_std"):
            actor_state.update({f"log_stds.{i}": actor_state["log_std"].clone() for i in range(len(policy.agent_ids))})
        policy.actor.load_state_dict({**policy.actor.state_dict(), **actor_state}, strict=True)
        policy._lora_ready = True
    else:
        for actor, state in prepared:
            actor.load_state_dict(state, strict=True)
    if critic_state is not None:
        policy.critic.load_state_dict(critic_state, strict=True)
    if policy.optimizer is not None:
        policy.set_training_progress(0)
    policy.source_checkpoint = {**checkpoint["_identity"], "algorithm": "ppo", "scope": scope,
                                "provenance": stored["provenance"]}


def capture_rng(cuda_device=None):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"generator": numpy[0],
            "keys": torch.tensor(numpy[1].astype(np.int64)), "position": numpy[2],
            "has_gauss": numpy[3], "cached_gaussian": numpy[4]},
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if cuda_device is None and torch.cuda.is_initialized() else [],
            **({"cuda_local": {"device": torch.device(cuda_device).index,
                               "state": torch.cuda.get_rng_state(cuda_device)}}
               if cuda_device is not None and torch.device(cuda_device).type == "cuda" else {})}


def restore_rng(state):
    numpy = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((numpy["generator"], numpy["keys"].numpy().astype(np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.random.set_rng_state(state["torch"].cpu())
    if "cuda_local" in state:
        local = state["cuda_local"]
        if not torch.cuda.is_available() or torch.cuda.current_device() != local["device"]:
            raise ValueError("Resume requires the checkpoint rank's local CUDA device")
        torch.cuda.set_rng_state(local["state"], local["device"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the checkpoint's CUDA RNG device topology")
        torch.cuda.set_rng_state_all(state["cuda"])


def optimization_contract(policy):
    names = ("lr", "lr_end", "lr_schedule", "target_kl", "gamma", "gae_lambda", "clip_range",
             "ent_coef", "vf_coef", "max_grad_norm", "n_steps", "n_epochs", "batch_size",
             "optimizer_name", "optimizer_kwargs", "scheduler_kwargs", "scheduler_clock", "loss", "critic_loss")
    return {**{key: getattr(policy, key) for key in names},
            "min_rollout_steps": getattr(policy, "min_rollout_steps", 1),
            **({key: getattr(policy, key) for key in ("reward_mode", "team_return_mode", "team_reward_reduction")}
               if hasattr(policy, "reward_mode") else {})}


def _resume_optimization_contract(state, algorithm):
    """Interpret original v1 contracts as hardcoded Adam/L2, constant or linear.

    Partially populated new contracts are rejected, rather than defaulted.
    Transfer/model-only loading does not compare training optimization settings.
    """
    from agents.common.optimization import OBJECTIVES, resolve_optimizer

    stored = state.get("optimization_contract")
    if not isinstance(stored, dict):
        raise ValueError("Checkpoint missing optimization_contract")
    added = {"optimizer_name", "optimizer_kwargs", "scheduler_kwargs", "scheduler_clock", "loss", "critic_loss"}
    present = added & stored.keys()
    legacy = not present
    if present and present != added:
        raise ValueError("Checkpoint has an incomplete optimization_contract")
    if legacy:
        if stored.get("lr_schedule") not in {"constant", "linear"}:
            raise ValueError("Legacy checkpoint schedule cannot be unambiguously interpreted")
        name, kwargs = resolve_optimizer()
        stored = {**stored, "optimizer_name": name, "optimizer_kwargs": kwargs,
                  "scheduler_kwargs": {}, "scheduler_clock": "training_budget_progress",
                  "loss": OBJECTIVES[algorithm], "critic_loss": "l2"}
    return stored, legacy


def resume_training(policy, path):
    checkpoint = _payload(path)
    state = checkpoint["training"]
    if not isinstance(state, dict) or state.get("boundary") != RESUME_BOUNDARY:
        raise ValueError("Checkpoint is model-only or has an unsupported resume boundary; use latest.pt")
    _validate_models(policy, checkpoint, critic=True)
    contract, legacy = _resume_optimization_contract(state, policy.algorithm)
    validate_checkpoint_compatibility({"optimization_contract": contract}, {
        "optimization_contract": optimization_contract(policy)})
    validate_checkpoint_compatibility(checkpoint["metadata"]["versions"], policy_metadata(policy)["versions"])
    if policy.optimizer is None:
        raise ValueError("Training resume requires a training learner")
    required = {"optimizer", "schedule_progress", "progress", "budget", "rng", "hooks", "recovery_seed"}
    if not required <= state.keys():
        raise ValueError("Checkpoint missing required recovery state")
    if not legacy and "scheduler" not in state:
        raise ValueError("Checkpoint missing required scheduler recovery state")
    scheduler_state = state.get("scheduler")
    if policy.scheduler is not None:
        if not isinstance(scheduler_state, dict) or set(scheduler_state) != set(policy.scheduler.state_dict()):
            raise ValueError("Checkpoint missing or incompatible native scheduler state")
        validate_checkpoint_compatibility(scheduler_state, {
            key: value for key, value in policy.scheduler_kwargs.items()})
        rates = [group["lr"] for group in state["optimizer"]["param_groups"]]
        validate_checkpoint_compatibility(scheduler_state, {"_last_lr": rates, "base_lrs": [policy.lr]})
    elif scheduler_state is not None:
        raise ValueError("Checkpoint has unexpected scheduler state")
    policy.actor.load_state_dict(checkpoint["models"]["actor"], strict=True)
    policy.critic.load_state_dict(checkpoint["models"]["critic"], strict=True)
    policy.optimizer.load_state_dict(state["optimizer"])
    if policy.scheduler is not None:
        policy.scheduler.load_state_dict(scheduler_state)
    policy.set_training_progress(state["schedule_progress"])
    policy.source_checkpoint = checkpoint["metadata"]["adaptation"]["source_checkpoint"]
    if hasattr(policy, "_lora_ready"):
        policy._lora_ready = True
    # Restore RNG after task/collector construction, which may consume draws.
    return state, checkpoint["metadata"]
