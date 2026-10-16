"""Small sweep specifications, persisted trial plans, and native selection ranking."""
from copy import deepcopy
import hashlib
import json
import math
import pickle
from pathlib import Path
import re

import yaml


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
                "wandb", "allow_duplicates"},
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
        if (not isinstance(path, str) or not re.fullmatch(r"(?:algorithm|network)\.[A-Za-z_][A-Za-z0-9_.]*", path)
                or path in {"algorithm.name", "algorithm.backend", "algorithm.params.progress_unit", "algorithm.params.update_version"}):
            raise ValueError(f"Search paths must name existing algorithm/network hyperparameters: {path!r}")
        domain_description(domain, search["method"], spec["allow_duplicates"])
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
            if type(value) not in {str, bool, int, float, type(None)} or isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Search choices must be finite YAML scalars")
            if kind == "int" and type(value) is not int:
                raise ValueError("Integer choices require integers")
        if not allow_duplicates and len(set(values)) != len(values):
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
                "wandb", "evaluation", "algorithm.name", "algorithm.backend", "adaptation")
    for override in OverridesParser.create().parse_overrides(spec["fixed_overrides"]):
        path = override.key_or_group
        if (override.is_sweep_override() or override.is_delete() or "/" in path or "@" in path
                or any(path == p or path.startswith(p + ".") or p.startswith(path + ".") for p in (*reserved, *seen))):
            raise ValueError(f"Conflicting or unsupported fixed override: {override.input_line}")
        seen.append(path)


def generate_plan(spec, tune):
    """Resolve native Ray domains once, then cross parameter points with seeds."""
    import numpy as np
    from ray.tune.search.variant_generator import generate_variants
    method = spec["search"]["method"]
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
            plan.append({"configuration_id": digest(parameters)[:16], "sample_index": index,
                         "replicate": replicate, "seed": seed, "parameters": parameters})
    return plan


def trial_settings(settings, entry):
    if entry is None:
        return settings
    result = deepcopy(settings)
    result.pop("trial_plan", None)
    result["seed"] = entry["seed"]
    result["sweep_trial"] = entry
    result["overrides"] = [*settings["overrides"], *(f"{k}={json.dumps(v)}" for k, v in entry["parameters"].items())]
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


def preflight(settings, command_builder, repo):
    """Use the native CLI/Hydra validator, freeze queued trials and eval protocol."""
    from application.cli import parse_args, apply_cli_overrides
    from application.configuration import validate_experiment_scenario
    from core.scenario import load_and_expand_scenario
    from training.evaluation_config import resolve_evaluation_protocol
    from training.algorithms import resolve_training_params
    from omegaconf import OmegaConf
    from hydra.core.override_parser.overrides_parser import OverridesParser

    inputs = Path(settings["root"]) / "sweep_inputs"
    original = load_and_expand_scenario(settings["scenarios"][settings["mode"]], validate=False)
    for override in OverridesParser.create().parse_overrides(settings["overrides"]):
        reference = select_path(original, override.key_or_group)
        if override.key_or_group.startswith("algorithm.params.") and not isinstance(reference, (dict, list)):
            validate_value(override.key_or_group, override.value(), reference)
    identity = {"algorithm": settings["mode"], "trial_id": "preflight", "run_id": "preflight", "wandb_id": "preflight"}

    def resolve(effective):
        args = parse_args(command_builder(effective, identity, inputs)[2:])
        return apply_cli_overrides(load_and_expand_scenario(args.scenario, validate=False, overrides=args.parameter_overrides), args)

    # Validate every categorical value and both range ends, even if a random
    # draw would miss an invalid option. Cross-parameter interactions are then
    # validated for every actual planned trial below.
    spec = settings["sweep"]
    for path, domain in spec["search"]["parameters"].items():
        reference = select_path(original, path)
        if isinstance(reference, (dict, list)):
            raise ValueError(f"Search path must be a scalar: {path}")
        values = domain_description(domain, spec["search"]["method"], spec["allow_duplicates"])
        candidates = values if isinstance(values, list) else (
            [domain["low"], domain["high"] - 1] if domain["type"] == "int" else
            [domain["low"], math.nextafter(domain["high"], domain["low"])])
        for value in candidates:
            validate_value(path, value, reference)
            probe = deepcopy(settings["trial_plan"][0])
            probe["parameters"] = {path: value}
            config = resolve(trial_settings(settings, probe))
            validate_experiment_scenario(config)
    # No experiment manifest exists yet; a corrected preflight may replace
    # inputs left by a previous validation failure before any trial launched.
    inputs.mkdir(exist_ok=True)
    protocol = None
    for index, entry in enumerate(settings["trial_plan"]):
        effective = trial_settings(settings, entry)
        try:
            config = resolve(effective)
            if config["algorithm"]["name"] != settings["mode"]:
                raise ValueError("Scenario algorithm differs from sweep algorithm")
            for path, value in entry["parameters"].items():
                reference = select_path(config, path)
                validate_value(path, value, select_path(original, path))
                if reference != value:
                    raise ValueError(f"Sampled value was not applied: {path}")
            validate_experiment_scenario(config)
            # Catch materialization/per-agent overrides shadowing searched params.
            for agent in config["agents"].values():
                if agent["algorithm"] == settings["mode"]:
                    params = resolve_training_params(agent, config)
                    for path, value in entry["parameters"].items():
                        if path.startswith("algorithm.params.") and select_path(params, path.removeprefix("algorithm.params.")) != value:
                            raise ValueError(f"Sampled path is shadowed by another override: {path}")
            selection = resolve_evaluation_protocol(config, "selection")
            final = resolve_evaluation_protocol(config, "final")
            if protocol is None:
                protocol = {"selection": selection, "final": final, "evaluation": config["evaluation"]}
            elif protocol != {"selection": selection, "final": final, "evaluation": config["evaluation"]}:
                raise ValueError("All sweep trials must use identical selection and held-out final protocols")
            if not config["evaluation"]["enabled"] or config["experiment"].get("distributed"):
                raise ValueError("Sweeps require selection evaluation and independent, non-DDP trainers")
            # Fixed overrides must not smuggle a Tune search expression into Hydra.
            entry["schedule"] = {k: config["training_defaults"].get(k) for k in ("lr_schedule", "learning_rate_end", "lr_scheduler_kwargs")}
            path = inputs / f"{index:06d}.yaml"
            OmegaConf.save(OmegaConf.create(config), path)
            entry["input_path"] = str(path)
            entry["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception as exc:
            raise ValueError(f"Invalid sweep trial {index}, seed={entry['seed']}, parameters={entry['parameters']}: {exc}") from exc
    settings["evaluation_protocol"] = protocol


def ranking(results):
    """Rank completed trials by their actual native best.pt selection records."""
    from agents.common.checkpoints import read_checkpoint
    from training.hooks import EvaluationCheckpointHook

    ranked, diagnostics = [], []
    for result in results:
        row = dict(result)
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
                    or any(select_path(config, path) != value for path, value in entry["parameters"].items())):
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
            row.update(best_checkpoint=str(best), best_evaluation={k: v for k, v in record.items() if not isinstance(v, (dict, list))},
                       selection_score=encoded, ranking_status="ranked", unavailable_tie_breaker=any(v is None for v in encoded))
            ranked.append((score, row))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            row.update(ranking_status="unranked", ranking_reason=str(exc))
            diagnostics.append(row)
    ranked.sort(key=lambda item: item[0], reverse=True)
    rows, previous, rank = [], None, 0
    for index, (score, row) in enumerate(ranked, 1):
        if score != previous:
            rank = index
        rows.append({"rank": rank, **row})
        previous = score
    return {"ranked_trials": rows, "diagnostics": diagnostics}
