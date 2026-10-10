"""Network metadata validation and policy-state restoration."""
from utils.torch_io import validate_checkpoint_compatibility


def validate_network_checkpoint(checkpoint, config, *, actor_only=False):
    """Old checkpoints imply MLP; new checkpoints declare their architecture."""
    validate_checkpoint_compatibility(checkpoint, {})
    fields = ("architecture", "actor_hidden_dims", "activation") if actor_only else tuple(config)
    if "network" in checkpoint:
        stored = checkpoint["network"]
        if not isinstance(stored, dict):
            raise ValueError("Checkpoint network metadata must be a mapping")
        validate_checkpoint_compatibility(stored, {key: config[key] for key in fields})
    else:
        validate_checkpoint_compatibility({"architecture": "mlp"}, {"architecture": config["architecture"]})
        # Very old PPO states can omit metadata. Tensor keys/shapes still validate.
        validate_checkpoint_compatibility(checkpoint, {
            key: config[key] for key in fields if key != "architecture" and key in checkpoint
        })


def transfer_network_state(state, expected, config, *, source_width=None, expand_inputs=False):
    """Keep architecture-specific tensor transformations outside policy code."""
    if config["architecture"] == "mlp":
        from agents.common.legacy_mlp_transfer import mlp_transfer_state
        return mlp_transfer_state(state, expected, source_width=source_width, expand_inputs=expand_inputs)
    if expand_inputs:
        raise ValueError(f"Observation extension is unsupported for {config['architecture']!r}")
    return state


def restore_policy_state(checkpoint, actor, critic, optimizer, *, actor_state=None, load_optimizer=True):
    """Validate both modules before loading; preserve the optimizer layout."""
    actor_state = checkpoint.get("actor") if actor_state is None else actor_state
    validate_checkpoint_compatibility(checkpoint, {}, states=[
        ("actor", actor_state, actor.state_dict()),
        ("critic", checkpoint.get("critic"), critic.state_dict()),
    ])
    actor.load_state_dict(actor_state, strict=True)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    if load_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
