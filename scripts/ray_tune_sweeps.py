"""Small sweep specifications, persisted trial plans, and native selection ranking."""
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
import pickle
from pathlib import Path
import re

import yaml

from ray_tune_sources import (NETWORKS, ADAPTATIONS, configuration_digest, read_sources, verify_source, inspect_model)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("experiment_id must contain only letters, digits, underscores or hyphens")
    return value


def keys(value, allowed, required=(), where="specification"):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= set(value):
        raise ValueError(f"Invalid {where}: allowed keys {sorted(allowed)}; required {sorted(required)}")


def positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def read_spec(path, repo):
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key, value in node.value:
            key = loader.construct_object(key)
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = loader.construct_object(value, deep=True)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    spec = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    keys(spec, {"name", "experiment_id", "algorithm", "scenario", "total_steps", "seeds", "sweep_seed",
                "fixed_overrides", "search", "max_concurrent_trials", "resources_per_trial", "evaluation",
                "wandb", "allow_duplicates", "source_manifest", "pretraining_seeds"},
         {"name", "experiment_id", "algorithm", "scenario", "total_steps", "seeds", "sweep_seed", "search"})
    identifier(spec["experiment_id"])
    if not isinstance(spec["name"], str) or not spec["name"].strip():
        raise ValueError("name must be a nonempty string")
    if spec["algorithm"] not in {"ppo", "mappo"}:
        raise ValueError("algorithm must be ppo or mappo")
    if not isinstance(spec["scenario"], str) or not (repo / spec["scenario"]).is_file():
        raise ValueError("scenario must name an existing YAML path (relative to the repository)")
    spec["scenario"] = str((repo / spec["scenario"]).resolve())
    positive(spec["total_steps"], "total_steps")
    positive(spec["sweep_seed"], "sweep_seed", 0)
    if not isinstance(spec["seeds"], list) or not spec["seeds"]:
        raise ValueError("seeds must be a nonempty list")
    for seed in spec["seeds"]:
        positive(seed, "training seed", 0)
        if seed >= 2**32:
            raise ValueError("training seed exceeds NumPy seed range")
    spec.setdefault("allow_duplicates", False)
    if type(spec["allow_duplicates"]) is not bool:
        raise ValueError("allow_duplicates must be boolean")
    if not spec["allow_duplicates"] and len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Duplicate training seeds require allow_duplicates: true")
    spec.setdefault("fixed_overrides", [])
    if not isinstance(spec["fixed_overrides"], list) or any(not isinstance(v, str) for v in spec["fixed_overrides"]):
        raise ValueError("fixed_overrides must be a list of Hydra override strings")
    spec.setdefault("max_concurrent_trials", 2)
    positive(spec["max_concurrent_trials"], "max_concurrent_trials")
    spec.setdefault("resources_per_trial", {"cpu": 6, "gpu": 1})
    resources = spec["resources_per_trial"]
    keys(resources, {"cpu", "gpu"}, {"cpu", "gpu"}, "resources_per_trial")
    positive(resources["cpu"], "resources_per_trial.cpu")
    if type(resources["gpu"]) is not int or resources["gpu"] not in {0, 1}:
        raise ValueError("resources_per_trial.gpu must be 0 or 1; multi-GPU/DDP trials are unsupported")
    spec.setdefault("evaluation", {})
    keys(spec["evaluation"], {"seed", "episodes", "every_steps", "max_steps"}, where="evaluation")
    spec.setdefault("wandb", {})
    keys(spec["wandb"], {"mode", "project", "entity", "group"}, where="wandb")
    if spec["wandb"].get("mode", "online") not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline or disabled")
    for key in ("project", "entity", "group"):
        if key in spec["wandb"] and (not isinstance(spec["wandb"][key], str) or not spec["wandb"][key]):
            raise ValueError(f"wandb.{key} must be a nonempty string")
    search = spec["search"]
    keys(search, {"method", "parameters", "num_samples"}, {"method", "parameters"}, "search")
    if search["method"] not in {"grid", "random"} or not isinstance(search["parameters"], dict) or not search["parameters"]:
        raise ValueError("search requires method grid/random and a nonempty parameters mapping")
    if search["method"] == "grid" and "num_samples" in search:
        raise ValueError("num_samples is only supported for random search")
    if search["method"] == "random":
        positive(search.get("num_samples"), "search.num_samples")
    # Fail before importing Ray or connecting to the cluster.
    for path, domain in search["parameters"].items():
        if (not isinstance(path, str) or not (path in {"network", "adaptation"} or re.fullmatch(r"(?:algorithm|network|adaptation\.lora)\.[A-Za-z_][A-Za-z0-9_.]*", path))
                or path in {"algorithm.name", "algorithm.backend", "algorithm.params.progress_unit", "algorithm.params.update_version"}):
            raise ValueError(f"Search paths must name existing algorithm/network hyperparameters: {path!r}")
        domain_description(domain, search["method"], spec["allow_duplicates"])
        if path in {"network", "adaptation"}:
            allowed = NETWORKS if path == "network" else ADAPTATIONS
            if domain["type"] != "choice" or any(not isinstance(v, str) or v not in allowed for v in domain.get("values", [])):
                raise ValueError(f"Unknown {path} preset; select choice values from {sorted(allowed)}")
    if "source_manifest" in spec:
        if not isinstance(spec["source_manifest"], str) or not spec["source_manifest"].strip():
            raise ValueError("source_manifest must name a YAML file")
        spec["source_manifest"] = str((repo / Path(spec["source_manifest"]).expanduser()).resolve())
        if spec["algorithm"] != "mappo":
            raise ValueError("source_manifest is only supported for MAPPO adaptation")
        if not isinstance(spec.get("pretraining_seeds"), list) or not spec["pretraining_seeds"]:
            raise ValueError("source_manifest requires explicit pretraining_seeds (distinct from MAPPO seeds)")
        for seed in spec["pretraining_seeds"]:
            positive(seed, "pretraining seed", 0)
            if seed >= 2**32:
                raise ValueError("pretraining seed exceeds NumPy seed range")
        if len(set(spec["pretraining_seeds"])) != len(spec["pretraining_seeds"]):
            raise ValueError("Duplicate pretraining seeds")
    elif "pretraining_seeds" in spec:
        raise ValueError("pretraining_seeds requires source_manifest")
    validate_override_conflicts(spec)
    return spec


def domain_description(domain, method, allow_duplicates=False):
    keys(domain, {"type", "values", "low", "high"}, {"type"}, "search domain")
    kind = domain["type"]
    if kind not in {"choice", "int", "uniform", "loguniform"}:
        raise ValueError(f"Unknown search-space type: {kind!r}")
    if "values" in domain:
        if kind not in {"choice", "int"} or set(domain) != {"type", "values"}:
            raise ValueError("Only choice/int domains accept values")
        values = domain["values"]
        if not isinstance(values, list) or not values:
            raise ValueError("values must be a nonempty list")
        for value in values:
            items = value if isinstance(value, list) else [value]
            if any(type(item) not in {str, bool, int, float, type(None)}
                   or isinstance(item, float) and not math.isfinite(item) for item in items):
                raise ValueError("Search choices must be finite YAML scalars or lists of scalars")
            if kind == "int" and type(value) is not int:
                raise ValueError("Integer choices require integers")
        if not allow_duplicates and len({digest(v) for v in values}) != len(values):
            raise ValueError("Duplicate choices require allow_duplicates: true")
        return values
    if kind == "choice" or set(domain) != {"type", "low", "high"}:
        raise ValueError("choice requires values; ranges require low and high")
    low, high = domain["low"], domain["high"]
    if (type(low) not in {int, float} or type(high) not in {int, float}
            or not math.isfinite(low) or not math.isfinite(high) or low >= high):
        raise ValueError("Search bounds must be finite numbers with low < high")
    if kind == "int":
        if type(low) is not int or type(high) is not int:
            raise ValueError("Integer range bounds must be integers")
        return range(low, high)  # Ray randint's upper bound is exclusive.
    if kind == "loguniform" and low <= 0:
        raise ValueError("loguniform.low must be positive")
    if method == "grid":
        raise ValueError("Grid search requires finite choices or integer ranges")
    return None


def validate_override_conflicts(spec):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    seen = list(spec["search"]["parameters"])
    reserved = ("experiment.name", "experiment.seed", "experiment.total_steps", "experiment.distributed",
                "experiment.torch_threads", "environment.render",
                "execution.experiment", "training_defaults", "agents",
                "checkpoint.resume", "execution.params.device", "execution.params.collection_device",
                "wandb", "evaluation", "algorithm.name", "algorithm.backend",
                "adaptation.checkpoint", "paths.pretrained_checkpoint")
    for override in OverridesParser.create().parse_overrides(spec["fixed_overrides"]):
        path = override.key_or_group
        if (override.is_sweep_override() or override.is_delete() or "/" in path or "@" in path
                or any(path == p or path.startswith(p + ".")
                       or path not in {"network", "adaptation"} and p.startswith(path + ".") for p in reserved)
                or any(path == p or (path not in {"network", "adaptation"} and p not in {"network", "adaptation"}
                                    and (path.startswith(p + ".") or p.startswith(path + "."))) for p in seen)):
            raise ValueError(f"Conflicting or unsupported fixed override: {override.input_line}")
        seen.append(path)


def generate_plan(spec, tune):
    """Resolve native Ray domains once, then cross parameter points with seeds."""
    import numpy as np
    from ray.tune.search.variant_generator import generate_variants
    method = spec["search"]["method"]
    default_arm = fixed_groups(spec).get("adaptation", "scratch")
    if "source_manifest" in spec and "adaptation" not in spec["search"]["parameters"] and "adaptation" not in fixed_groups(spec):
        from core.scenario import load_and_expand_scenario
        default_arm = load_and_expand_scenario(spec["scenario"], validate=False)["adaptation"]["mode"]
    domains = {}
    cardinality = 1
    for path, domain in sorted(spec["search"]["parameters"].items()):
        values = domain_description(domain, method, spec["allow_duplicates"])
        if method == "grid":
            domains[path] = tune.grid_search(list(values))
        elif "values" in domain:
            domains[path] = tune.choice(values)
        else:
            domains[path] = getattr(tune, {"int": "randint"}.get(domain["type"], domain["type"]))(domain["low"], domain["high"])
        cardinality *= len(values) if values is not None else math.inf
    count = cardinality if method == "grid" else spec["search"]["num_samples"]
    if not spec["allow_duplicates"] and count > cardinality:
        raise ValueError(f"Requested {count} distinct samples from only {cardinality} configurations")
    random_state = np.random.default_rng(spec["sweep_seed"])
    points, seen = [], set()
    attempts = 0
    while len(points) < count:
        for _, resolved in generate_variants({"config": domains}, random_state=random_state):
            parameters = resolved["config"]
            key = digest(parameters)
            if spec["allow_duplicates"] or key not in seen:
                points.append(parameters)
                seen.add(key)
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("Could not draw enough distinct samples; reduce num_samples or allow duplicates")
    plan = []
    for index, parameters in enumerate(points):
        for replicate, seed in enumerate(spec["seeds"]):
            arm = parameters.get("adaptation", default_arm)
            sources = spec.get("pretraining_seeds", [None]) if arm != "scratch" else [None]
            for pretraining_seed in sources:
                entry = {"configuration_id": digest(parameters)[:16], "sample_index": index,
                         "replicate": replicate, "seed": seed, "parameters": deepcopy(parameters)}
                if "source_manifest" in spec:
                    entry["pretraining_seed"] = pretraining_seed
                plan.append(entry)
    return plan


def fixed_groups(spec):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    return {o.key_or_group: o.value() for o in OverridesParser.create().parse_overrides(spec.get("fixed_overrides", []))
            if o.key_or_group in {"network", "adaptation"}}


def uses_model_groups(spec):
    return bool(spec and ("source_manifest" in spec
                or {"network", "adaptation"} & spec["search"].get("parameters", {}).keys()
                or fixed_groups(spec)))


def sampled_value(config, path, entry):
    if path == "network":
        return entry.get("network_preset", config.get("_configuration", {}).get("choices", {}).get("network"))
    if path == "adaptation":
        return config["adaptation"]["mode"]
    return select_path(config, path)


def trial_settings(settings, entry):
    if entry is None:
        return settings
    result = deepcopy(settings)
    result.pop("trial_plan", None)
    result["seed"] = entry["seed"]
    result["sweep_trial"] = entry
    result["overrides"] = [*settings["overrides"], *(f"{k}={json.dumps(v)}" for k, v in entry["parameters"].items())]
    if entry.get("source"):
        result["overrides"].append(f"adaptation.checkpoint={json.dumps(entry['source']['checkpoint'])}")
    return result


def select_path(config, path):
    value = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Unknown configuration path: {path}")
        value = value[part]
    return value


def validate_value(path, value, reference):
    if reference is None:
        if value is not None and type(value) not in {int, float}:
            raise ValueError(f"{path} requires a number or null")
    elif type(reference) is float:
        if type(value) not in {int, float}:
            raise ValueError(f"{path} requires a number")
    elif type(value) is not type(reference):
        raise ValueError(f"{path} requires {type(reference).__name__}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    # These checks run in learner constructors rather than the native CLI's
    # config-only path. Catch them here before allocating any Tune workers.
    name = path.rsplit(".", 1)[1]
    if name in {"n_steps", "n_epochs", "batch_size", "frames_per_batch", "min_rollout_steps"} and value < 1:
        raise ValueError(f"{path} must be positive")
    if name in {"gamma", "gae_lambda"} and not 0 <= value <= 1:
        raise ValueError(f"{path} must be between 0 and 1")


def validate_search_effect(config, parameters):
    """Check actual learner inputs, including ignored common/per-role fallbacks."""
    from agents.common.networks import resolve_network_config, model_spec
    from training.algorithms import resolve_training_params
    for agent in config["agents"].values():
        if agent["algorithm"] != config["algorithm"]["name"]:
            continue
        params = resolve_training_params(agent, config)
        network = resolve_network_config(params, default_hidden_dims=[256, 256])
        for path, value in parameters.items():
            if path.startswith("algorithm.params."):
                actual = select_path(params, path.removeprefix("algorithm.params."))
            elif path.startswith("adaptation.lora."):
                actual = select_path(params, path.removeprefix("adaptation."))
            elif path.startswith("network."):
                field = path.removeprefix("network.")
                roles = [field.split(".")[0]] if field.startswith(("actor.", "critic.")) else ["actor", "critic"]
                actuals = []
                for role in roles:
                    spec = model_spec(network, role)
                    key = field.removeprefix(role + ".")
                    if key == f"{role}_hidden_dims":
                        key = "hidden_dims"
                    elif key in {"actor_hidden_dims", "critic_hidden_dims"}:
                        continue
                    if key == "encoder" and isinstance(value, str):
                        key = "encoder.type"
                    try:
                        actuals.append(select_path(spec, key))
                    except ValueError:
                        pass
                if value not in actuals:
                    raise ValueError(f"Sampled path is ineffective or shadowed: {path}")
                continue
            else:
                continue
            if actual != value:
                raise ValueError(f"Sampled path is shadowed by another override: {path}")


def preflight(settings, command_builder, repo):
    """Compose every categorical combination, validate native models, freeze inputs.

    Uses temporary inputs in --preflight-only. Never overwrite an existing
    experiment manifest/configuration, even when invoked outside the CLI.
    """
    from application.cli import parse_args, apply_cli_overrides
    from application.configuration import validate_experiment_scenario
    from core.scenario import load_and_expand_scenario
    from training.evaluation_config import resolve_evaluation_protocol
    from training.configuration import PROFILING_DEFAULTS
    from omegaconf import OmegaConf
    from hydra.core.override_parser.overrides_parser import OverridesParser

    inputs = Path(settings["root"]) / "sweep_inputs"
    if (Path(settings["root"]) / "experiment.json").exists():
        raise ValueError("Preflight cannot overwrite an existing experiment; use --preflight-only or --restore")
    spec = settings["sweep"]
    original = load_and_expand_scenario(settings["scenarios"][settings["mode"]], validate=False)
    optional = set()
    for override in OverridesParser.create().parse_overrides(settings["overrides"]):
        path = override.key_or_group
        if path in {"network", "adaptation"} or path.startswith(("network.", "adaptation.")):
            continue  # Checked against the selected Hydra preset below.
        profile_key = path.removeprefix("experiment.") if path.startswith("experiment.") else None
        try:
            reference = select_path(original, path)
        except ValueError:
            if profile_key not in PROFILING_DEFAULTS:
                raise
            reference = PROFILING_DEFAULTS[profile_key]
            optional.add(path)
        if profile_key in PROFILING_DEFAULTS:
            reference = PROFILING_DEFAULTS[profile_key]
        if profile_key in PROFILING_DEFAULTS or path.startswith("algorithm.params.") and not isinstance(reference, (dict, list)):
            validate_value(path, override.value(), reference)
    identity = {"algorithm": settings["mode"], "trial_id": "preflight", "run_id": "preflight", "wandb_id": "preflight"}
    resolution_cache = {}

    def resolve(entry):
        # Never reuse a previous preflight's frozen path during validation.
        clean = {k: v for k, v in entry.items() if k not in {"input_path", "input_sha256"}}
        command = command_builder(trial_settings(settings, clean), identity, inputs)
        cache_key = digest(command)
        if cache_key in resolution_cache:
            return deepcopy(resolution_cache[cache_key])
        args = parse_args(command[2:])
        overrides = ["++" + value.lstrip("+") if value.lstrip("+").split("=", 1)[0] in optional else value
                     for value in args.parameter_overrides]
        try:
            config = apply_cli_overrides(load_and_expand_scenario(args.scenario, validate=False, overrides=overrides), args)
            resolution_cache[cache_key] = config
            return deepcopy(config)
        except Exception as exc:
            raise ValueError(f"Invalid Hydra sweep configuration: {exc}") from exc

    sources = read_sources(spec["source_manifest"]) if "source_manifest" in spec else {}
    if "source_manifest" in spec:
        settings["source_manifest_sha256"] = hashlib.sha256(Path(spec["source_manifest"]).read_bytes()).hexdigest()
    # Validate the Cartesian product of ALL categorical options and range ends,
    # including interactions missed by random sampling. Separate specs keep
    # architecture-specific spaces small and explicit.
    paths, options = [], []
    for path, domain in spec["search"]["parameters"].items():
        values = domain_description(domain, spec["search"]["method"], spec["allow_duplicates"])
        candidates = values if isinstance(values, list) else (
            [domain["low"], domain["high"] - 1] if domain["type"] == "int" else
            [domain["low"], math.nextafter(domain["high"], domain["low"])])
        paths.append(path)
        options.append(candidates)
    probes = []
    for values in product(*options):
        parameters = dict(zip(paths, values))
        arm = parameters.get("adaptation", fixed_groups(spec).get("adaptation", original["adaptation"]["mode"]))
        for seed in spec.get("pretraining_seeds", [None]) if arm != "scratch" else [None]:
            probes.append({**settings["trial_plan"][0], "parameters": parameters, "pretraining_seed": seed})
    # Aggregate missing sources across the entire matrix, rather than hiding
    # later missing architectures behind the first failure.
    missing = set()
    for entry in [*probes, *settings["trial_plan"]]:
        entry.pop("source", None)
        baseline = resolve({**entry, "parameters": {k: v for k, v in entry["parameters"].items() if k in {"network", "adaptation"}}})
        for path, value in entry["parameters"].items():
            if path not in {"network", "adaptation"}:
                validate_value(path, value, select_path(baseline, path))
        config = resolve(entry)
        preset = config["_configuration"]["choices"].get("network")
        entry.update(network_preset=preset, adaptation=config["adaptation"]["mode"])
        if entry["adaptation"] != "scratch":
            key = (preset, entry.get("pretraining_seed"))
            if settings["mode"] != "mappo" or key not in sources:
                missing.add(str(key))
            else:
                entry["source"] = deepcopy(sources[key])
        elif config["adaptation"]["checkpoint"] is not None:
            raise ValueError("Scratch trials cannot use pretrained sources")
    if missing:
        raise ValueError("Missing verified PPO sources for network/pretraining_seed: " + ", ".join(sorted(missing))
                         + ". Populate source_manifest using scripts/ray_tune_sources.py and real native PPO best.pt files.")
    for source in {digest(e["source"]): e["source"] for e in probes if e.get("source")}.values():
        verify_source(source)

    protocol, comparison, seen, model_cache = None, None, set(), {}

    def validate(entry):
        config = resolve(entry)
        if config["algorithm"]["name"] != settings["mode"]:
            raise ValueError("Scenario algorithm differs from sweep algorithm")
        baseline = resolve({**entry, "parameters": {k: v for k, v in entry["parameters"].items() if k in {"network", "adaptation"}}})
        for path, value in entry["parameters"].items():
            if path not in {"network", "adaptation"}:
                validate_value(path, value, select_path(baseline, path))
            if sampled_value(config, path, entry) != value:
                raise ValueError(f"Sampled value was not applied: {path}")
        validate_experiment_scenario(config)
        validate_search_effect(config, entry["parameters"])
        selection = resolve_evaluation_protocol(config, "selection")
        final = resolve_evaluation_protocol(config, "final")
        protocols = {"selection": selection, "final": final, "evaluation": config["evaluation"]}
        if not config["evaluation"]["enabled"] or config["experiment"].get("distributed"):
            raise ValueError("Sweeps require selection evaluation and independent, non-DDP trainers")
        # Architectures/arms may vary; observation, reward, maps and opponents
        # must stay controlled. Training seeds are replicated independently.
        common = {"environment": config["environment"], "maps": config["maps"],
                  "agents": {aid: {k: v for k, v in agent.items() if k != "params"}
                             for aid, agent in config["agents"].items()}, "total_steps": config["experiment"]["total_steps"]}
        model_key = digest({"network": config["network"], "adaptation": config["adaptation"], "common": common})
        if model_key not in model_cache:
            model_cache[model_key] = inspect_model(config, entry.get("source"))
        return config, protocols, common, model_cache[model_key]

    for entry in probes:
        try:
            validate(entry)
        except Exception as exc:
            raise ValueError(f"Invalid categorical combination {entry['parameters']}: {exc}") from exc
    resolved = []
    for index, entry in enumerate(settings["trial_plan"]):
        try:
            config, protocols, common, model = validate(entry)
            if protocol is not None and protocol != protocols:
                raise ValueError("All sweep trials must use identical selection and held-out final protocols")
            if comparison is not None and comparison != common:
                raise ValueError("All architecture/adaptation trials must use matching environment, rewards, maps and budgets")
            protocol, comparison = protocols, common
            entry["model"] = deepcopy(model)
            entry["config_digest"] = configuration_digest(config)
            entry["trial_key"] = digest({"experiment": settings["experiment_id"], "algorithm": settings["mode"],
                                         "preset": entry["network_preset"], "config": entry["config_digest"],
                                         "source": entry.get("source"), "seed": entry["seed"],
                                         "repeat": [entry["sample_index"], entry["replicate"]] if spec["allow_duplicates"] else None})
            if entry["config_digest"] in seen and not spec["allow_duplicates"]:
                raise ValueError("Duplicate effective trial configurations; remove shadowed/alias choices or allow_duplicates")
            seen.add(entry["config_digest"])
            entry["schedule"] = {k: config["training_defaults"].get(k) for k in ("lr_schedule", "learning_rate_end", "lr_scheduler_kwargs")}
            config["wandb"]["tune"].update(network_preset=entry["network_preset"], adaptation=entry["adaptation"],
                pretraining_seed=entry.get("pretraining_seed"), source_sha256=entry.get("source", {}).get("sha256"),
                source_checkpoint=entry.get("source", {}).get("checkpoint"), trial_key=entry["trial_key"],
                config_digest=entry["config_digest"], **model)
            if uses_model_groups(spec):
                config["wandb"]["group"] = settings["wandb"].get("group") or settings["experiment_id"] + (
                    "-pretraining" if settings["mode"] == "ppo" else "-adaptation")
            resolved.append((entry, config))
        except Exception as exc:
            raise ValueError(f"Invalid sweep trial {index}, seed={entry['seed']}, parameters={entry['parameters']}: {exc}") from exc
    inputs.mkdir(exist_ok=True)
    for index, (entry, config) in enumerate(resolved):
        path = inputs / f"{index:06d}.yaml"
        OmegaConf.save(OmegaConf.create(config), path)
        frozen = load_and_expand_scenario(str(path), validate=False)
        validate_experiment_scenario(frozen)
        if {k: v for k, v in frozen.items() if k != "_configuration"} != {k: v for k, v in config.items() if k != "_configuration"}:
            raise ValueError("Frozen configuration changes when reconstructed")
        entry.update(input_path=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    settings["evaluation_protocol"] = protocol


def ranking(results):
    """Rank completed trials by their actual native best.pt selection records."""
    from agents.common.checkpoints import read_checkpoint
    from training.hooks import EvaluationCheckpointHook

    ranked, diagnostics = [], []
    for result in results:
        row = dict(result)
        row.setdefault("parameter_counts", (row.get("model") or {}).get("parameter_counts", {}))
        source = row.get("source") or {}
        row.setdefault("source_checkpoint", source.get("checkpoint"))
        row.setdefault("source_sha256", source.get("sha256"))
        row["ranking_status"] = "incomplete_or_failed"
        if row.get("error") or row.get("status") != "completed":
            diagnostics.append(row)
            continue
        try:
            out = Path(row["training_output_dir"])
            best = out / "best.pt"
            if not best.is_file():
                raise ValueError("missing best.pt / selection evaluation")
            payload = read_checkpoint(best)
            metadata = payload["metadata"]
            record = metadata["selection"]
            if not isinstance(record, dict):
                raise ValueError("missing checkpoint selection record")
            strategy = "completion_progress" if row["algorithm"] == "ppo" else "team_completion"
            if record.get("selection_strategy") != strategy:
                raise ValueError("unexpected checkpoint selection strategy")
            required = (("completion_rate", "collision_rate", "mean_net_progress") if row["algorithm"] == "ppo" else
                        ("team_both_finished_rate", "team_completion_rate", "team_collision_rate"))
            if any(type(record.get(k)) not in {int, float} or not math.isfinite(record[k]) for k in required):
                raise ValueError("missing or nonfinite primary evaluation metrics")
            if row["algorithm"] == "mappo" and record["team_both_finished_rate"] < 1:
                if type(record.get("mean_net_progress")) not in {int, float} or not math.isfinite(record["mean_net_progress"]):
                    raise ValueError("missing or nonfinite net-progress tie-breaker")
            score = EvaluationCheckpointHook.selection_score(record, strategy)
            if any(math.isnan(v) or v == math.inf for v in score):
                raise ValueError("nonfinite evaluation selection score")
            encoded = [v if math.isfinite(v) else None for v in score]
            if record.get("selection_score") != encoded:
                raise ValueError("saved selection score differs from native ranking criteria")
            history = out / "evaluation_history.jsonl"
            found = False
            with history.open() as stream:
                for line in stream:
                    try:
                        candidate = json.loads(line)
                    except ValueError:
                        continue
                    if candidate == record:
                        found = True
                        break
            if not found:
                raise ValueError("best checkpoint selection record missing from evaluation history")
            identity = json.loads((out / "ray_trial.json").read_text())
            if metadata["provenance"].get("run_id") != identity["run_id"] or metadata["algorithm"] != row["algorithm"]:
                raise ValueError("best checkpoint trial identity differs")
            saved = identity["settings"]
            entry = saved["sweep_trial"]
            config = metadata["configuration"]
            if (config["experiment"]["seed"] != entry["seed"] or config["experiment"]["total_steps"] != saved["total_steps"]
                    or row["seed"] != entry["seed"] or row["hyperparameters"] != entry["parameters"]
                    or any(sampled_value(config, path, entry) != value for path, value in entry["parameters"].items())):
                raise ValueError("best checkpoint sweep configuration differs")
            expected = saved["evaluation_protocol"]["selection"]
            actual = record.get("evaluation_protocol", {})
            if not isinstance(actual, dict):
                raise ValueError("missing selection evaluation protocol")
            if (actual.get("name") != "selection"
                    or actual.get("seeds") != list(range(expected["seed"], expected["seed"] + expected["episodes"]))
                    or actual.get("max_steps") != expected["max_steps"] or actual.get("target_laps") != expected["target_laps"]
                    or record.get("episodes") != expected["episodes"]
                    or len(record.get("episode_results", [])) != expected["episodes"]):
                raise ValueError("incomplete evaluation or changed selection protocol; final-test results cannot rank trials")
            if entry.get("config_digest") and configuration_digest(config) != entry["config_digest"]:
                raise ValueError("best checkpoint frozen configuration differs")
            source = entry.get("source")
            if source and (metadata["adaptation"]["source_checkpoint"] or {}).get("sha256") != source["sha256"]:
                raise ValueError("best checkpoint PPO source differs")
            counts = metadata["provenance"].get("parameter_counts", entry.get("model", {}).get("parameter_counts", {}))
            if counts:
                counts = {**counts, "learner_trainable": counts["actor_trainable"] + counts["critic_trainable"]}
            protocol_id = digest({"algorithm": row["algorithm"], "protocol": saved["evaluation_protocol"],
                                  "maps": config.get("maps"), "environment": config.get("environment"),
                                  "agents": {aid: {k: v for k, v in agent.items() if k != "params"}
                                             for aid, agent in config.get("agents", {}).items()},
                                  "map_protocols": metadata["provenance"].get("map_protocols")})
            row.update(network_preset=entry.get("network_preset"), adaptation=entry.get("adaptation", config.get("adaptation", {}).get("mode")),
                       pretraining_seed=entry.get("pretraining_seed"), source_checkpoint=(source or {}).get("checkpoint"),
                       source_sha256=(source or {}).get("sha256"), parameter_counts=counts, protocol_id=protocol_id,
                       total_training_decisions=row.get("environment_steps"),
                       final_test_protocol=saved["evaluation_protocol"].get("final"))
            row.update(best_checkpoint=str(best), best_evaluation={k: v for k, v in record.items() if not isinstance(v, (dict, list))},
                       selection_score=encoded, ranking_status="ranked", unavailable_tie_breaker=any(v is None for v in encoded))
            ranked.append((score, row))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            row.update(ranking_status="unranked", ranking_reason=str(exc))
            diagnostics.append(row)
    # Each protocol has its own ranks. Never compare held-out final tests, or
    # combine results with different tracks, termination rules or horizons.
    partitions = {}
    for score, row in ranked:
        partitions.setdefault(row["protocol_id"], []).append((score, row))
    rows, comparison_groups = [], {}
    for protocol_id, partition in sorted(partitions.items()):
        partition.sort(key=lambda item: item[0], reverse=True)
        previous, rank = None, 0
        for index, (score, row) in enumerate(partition, 1):
            if score != previous:
                rank = index
            ranked_row = {"rank": rank, **row}
            rows.append(ranked_row)
            group = f"{protocol_id}:{row.get('network_preset')}:{row.get('adaptation')}"
            comparison_groups.setdefault(group, []).append(ranked_row)
            previous = score
    return {"ranked_trials": rows, "comparison_groups": comparison_groups, "diagnostics": diagnostics}
