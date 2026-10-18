"""Hydra composition and projection onto the existing task/learner fields."""
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def source_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def compose_configuration(path, overrides=None):
    """Compose defaults and standard Hydra overrides without changing cwd."""
    path = source_path(path)
    with initialize_config_dir(config_dir=str(path.parent), version_base="1.3"):
        config = compose(config_name=path.name, return_hydra_config=True, overrides=[
            f"hydra.searchpath=[file://{CONFIG_ROOT}]", *(overrides or ()),
        ])
    choices = OmegaConf.to_container(config.hydra.runtime.choices, resolve=True)
    OmegaConf.set_struct(config, False)
    config.pop("hydra")
    if "paths" in config:
        config.paths.project_root = str(PROJECT_ROOT)
    result = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a mapping")
    if "includes" in result:
        raise ValueError("'includes' was removed; use a Hydra defaults list")
    result["_configuration"] = {
        "source": str(path),
        "choices": {k: v for k, v in choices.items() if not k.startswith("hydra/")},
        "overrides": list(overrides or ()),
        "hydra_version": hydra.__version__,
    }
    return materialize_configuration(result)


def materialize_configuration(config):
    """Translate config groups into inputs accepted by the existing consumers."""
    if "environment" in config:
        environment = config["environment"]
        environment["map_dir"] = str(source_path(
            environment.get("map_dir") or environment.get("map_root") or "maps"))
    if "algorithm" not in config:
        return config
    algorithm, execution, adaptation = (
        config["algorithm"], config["execution"], config["adaptation"])
    if adaptation["checkpoint"] is not None:
        adaptation["checkpoint"] = str(source_path(adaptation["checkpoint"]))
    config["experiment"] = OmegaConf.to_container(OmegaConf.merge(
        execution["experiment"], config.get("experiment", {})), resolve=True)
    name = algorithm["name"]
    config["experiment"][f"{name}_backend"] = algorithm["backend"]
    explicit_params = config.get("training_defaults", {})
    config["training_defaults"] = OmegaConf.to_container(OmegaConf.merge(
        algorithm["params"], execution["params"], explicit_params), resolve=True)
    if name == "mappo":
        config["mappo"] = OmegaConf.to_container(OmegaConf.merge({
            "actor_mode": config["network"]["actor"]["parameter_sharing"],
            "reward_mode": "team_shared", "critic_mode": "shared_team",
            "team_reward_reduction": "mean",
        }, config.get("mappo", {})), resolve=True)
        config["network"]["actor"]["parameter_sharing"] = config["mappo"]["actor_mode"]
        config["training_defaults"].update(
            pretrained_actor_checkpoint=adaptation["checkpoint"],
            require_pretrained_actor=adaptation["mode"] != "scratch",
            pretrained_actor_observation_extension=adaptation["observation_extension"],
        )
        if adaptation["lora"] is not None:
            config["training_defaults"]["lora"] = adaptation["lora"]
    elif adaptation["checkpoint"] is not None:
        config["experiment"].setdefault("checkpoint", adaptation["checkpoint"])
        if adaptation["observation_extension"] is not None:
            config["training_defaults"]["pretrained_observation_extension"] = adaptation["observation_extension"]
    config["training_defaults"] = OmegaConf.to_container(OmegaConf.merge(
        config["training_defaults"], explicit_params), resolve=True)
    checkpoint = config["experiment"].get("checkpoint")
    if checkpoint is not None:
        config["experiment"]["checkpoint"] = str((
            Path(config["_configuration"]["source"]).parent / Path(checkpoint).expanduser()).resolve())
    if config.get("checkpoint", {}).get("resume") is not None:
        config["checkpoint"]["resume"] = str(source_path(config["checkpoint"]["resume"]))
    return config


def validate_configuration(config):
    """Reject choices for which the current learners have no implementation."""
    from core.scenario import ScenarioError

    if "algorithm" not in config:
        return
    algorithm, network, adaptation = config["algorithm"], config["network"], config["adaptation"]
    if algorithm["name"] not in {"ppo", "mappo"} or algorithm["backend"] != "torchrl":
        raise ScenarioError("Only TorchRL PPO and MAPPO are implemented")
    checkpoint = config.get("checkpoint", {})
    if set(checkpoint) - {"resume", "every_steps", "every_updates"}:
        raise ScenarioError("Unknown checkpoint setting")
    for key in ("every_steps", "every_updates"):
        value = checkpoint.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ScenarioError(f"checkpoint.{key} must be a positive integer or null")
    if checkpoint.get("resume") is not None and (not isinstance(checkpoint["resume"], str) or not checkpoint["resume"].strip()):
        raise ScenarioError("checkpoint.resume must be a path or null")
    expected_gae = "gae" if algorithm["name"] == "ppo" else "multi_agent_gae"
    if algorithm["advantage_estimator"] != expected_gae:
        raise ScenarioError("Unsupported advantage estimator for the selected algorithm")
    from agents.common.optimization import resolve_optimizer, validate_objective
    try:
        resolve_optimizer(algorithm["optimizer"], algorithm.get("optimizer_kwargs", {}))
        validate_objective(algorithm["name"], algorithm["loss"], algorithm["critic_loss"])
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc
    from agents.common.networks import resolve_network_config, model_spec
    try:
        resolved_network = resolve_network_config({"network": network,
            "lora": adaptation["lora"] or config["training_defaults"].get("lora")},
            default_hidden_dims=[256, 256])
        if (algorithm["name"] == "mappo"
                and model_spec(resolved_network, "critic")["encoder"]["type"] != "identity"):
            raise ValueError("MAPPO global-state critic has no compatible LiDAR layout; use an identity critic encoder")
        convolutional = any(model_spec(resolved_network, role)["encoder"]["type"] != "identity"
                            for role in ("actor", "critic"))
        if convolutional or adaptation["lora"] is not None:
            from wrappers.observations.composer import ObservationComposer
            from agents.common.observations import observation_segments
            layouts = []
            for agent in config["agents"].values():
                if agent.get("trainable"):
                    composer = ObservationComposer.from_config(agent["observation"], config["environment"])
                    if convolutional:
                        layouts.append(observation_segments(composer.contract, composer.obs_dim))
                    if adaptation["lora"] is not None:
                        from agents.common.lora import validate_lora_spec
                        from agents.common.encoders import fusion_feature_segments
                        actor_spec = model_spec(resolved_network, "actor")
                        width = composer.obs_dim
                        if actor_spec["encoder"]["type"] != "identity":
                            width = fusion_feature_segments(actor_spec["encoder"], layouts[-1])["neighbors"][1]
                        validate_lora_spec(actor_spec, adaptation["lora"], input_dim=width)
            if any(layout != layouts[0] for layout in layouts[1:]):
                raise ValueError("CNN actors require identical observation segments across learners")
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc
    actor = network.get("actor", {})
    critic = network.get("critic", {})
    if actor and actor.get("distribution", "tanh_normal") != "tanh_normal":
        raise ScenarioError("Only an actor with a tanh_normal distribution is implemented")
    expected_input = "local_observation" if algorithm["name"] == "ppo" else "global_state"
    expected_sharing = "independent" if algorithm["name"] == "ppo" else "shared_team"
    if critic and (critic.get("input_type") != expected_input
                   or critic.get("parameter_sharing") != expected_sharing):
        raise ScenarioError(f"{algorithm['name'].upper()} requires a critic with {expected_input} input and {expected_sharing} parameters")
    if actor and actor.get("parameter_sharing") not in {"independent", "shared"}:
        raise ScenarioError("Actor parameter_sharing must be independent or shared")
    if adaptation["mode"] not in {"scratch", "full_finetune", "lora"}:
        raise ScenarioError("Unsupported adaptation mode")
    expected_scope = "actor_and_critic" if algorithm["name"] == "ppo" else "actor_only"
    if adaptation["load_scope"] != expected_scope:
        raise ScenarioError(f"{algorithm['name'].upper()} adaptation requires {expected_scope} loading")
    if adaptation["mode"] != "scratch" and not adaptation["checkpoint"]:
        raise ScenarioError("Fine-tuning requires adaptation.checkpoint")
    if adaptation["mode"] == "scratch" and (adaptation["checkpoint"] or adaptation["lora"]):
        raise ScenarioError("Scratch adaptation cannot specify a checkpoint or LoRA")
    if adaptation["mode"] == "lora" and (algorithm["name"] != "mappo"
            or actor.get("parameter_sharing") != "shared" or not adaptation["lora"]):
        raise ScenarioError("LoRA requires MAPPO with a shared actor and per-agent adapters")
    if adaptation["mode"] != "lora" and adaptation["lora"] is not None:
        raise ScenarioError("LoRA settings require adaptation=lora")
    if adaptation["lora"] is not None:
        from agents.common.lora import resolve_lora_config
        try:
            lora = resolve_lora_config(adaptation["lora"])
        except ValueError as exc:
            raise ScenarioError(str(exc)) from exc
        if lora["mode"] != "per_agent":
            raise ScenarioError("LoRA requires per_agent adapters")
    for key in ("train", "selection", "final_test"):
        value = config["maps"].get(key)
        if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value):
            raise ScenarioError(f"maps.{key} must be a nonempty list of map bundles")
