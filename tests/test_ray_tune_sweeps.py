"""Focused sweep planning, scheduling/restoration and native ranking checks."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from test_ray_tune_trials import adapter
import ray_tune_sweeps as sweeps


@pytest.fixture
def tune(monkeypatch):
    # Lightweight stand-ins exercise orchestration without installing Ray.
    class Domain:
        def __init__(self, kind, *values):
            self.kind, self.values = kind, values

        def sample(self, rng):
            if self.kind == "choice":
                values = self.values[0]
                return values[int(rng.integers(len(values)))]
            low, high = self.values
            if self.kind == "randint":
                return int(rng.integers(low, high))
            if self.kind == "uniform":
                return float(rng.uniform(low, high))
            import numpy as np
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    def variants(spec, random_state):
        paths, options = [], []
        for path, value in spec["config"].items():
            paths.append(path)
            options.append(value["grid_search"] if isinstance(value, dict) else [value.sample(random_state)])
        for values in product(*options):
            yield {}, {"config": dict(zip(paths, values))}

    module = ModuleType("ray.tune.search.variant_generator")
    module.generate_variants = variants
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(grid_search=lambda values: {"grid_search": values},
                           **{kind: (lambda *v, kind=kind: Domain(kind, *v)) for kind in
                              ("choice", "randint", "uniform", "loguniform")})


def spec_file(tmp_path, **updates):
    spec = yaml.safe_load((adapter.REPO / "scripts/sweeps/ppo_lr_smoke.yaml").read_text())
    spec.update(updates)
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_grid_and_replicates_distinguish_configurations(tmp_path, tune):
    spec = sweeps.read_spec(spec_file(tmp_path, seeds=[42, 43]), adapter.REPO)
    plan = sweeps.generate_plan(spec, tune)
    assert len(plan) == 6
    assert len({p["configuration_id"] for p in plan}) == 3
    assert len({(p["configuration_id"], p["seed"]) for p in plan}) == 6
    assert plan == sweeps.generate_plan(spec, tune)
    spec["search"]["parameters"]["algorithm.params.n_epochs"] = {"type": "int", "low": 1, "high": 3}
    assert len(sweeps.generate_plan(spec, tune)) == 12


def test_random_bounds_reproducibility_and_distinct_samples(tmp_path, tune):
    search = {"method": "random", "num_samples": 5, "parameters": {
        "algorithm.params.learning_rate": {"type": "loguniform", "low": .0001, "high": .001},
        "algorithm.params.ent_coef": {"type": "uniform", "low": 0., "high": .1},
        "algorithm.params.n_epochs": {"type": "int", "low": 1, "high": 4},
        "algorithm.params.batch_size": {"type": "int", "values": [64, 128]}}}
    spec = sweeps.read_spec(spec_file(tmp_path, search=search, fixed_overrides=[]), adapter.REPO)
    a = sweeps.generate_plan(spec, tune)
    assert a == sweeps.generate_plan(spec, tune) and len(a) == 5
    for entry in a:
        p = entry["parameters"]
        assert .0001 <= p["algorithm.params.learning_rate"] < .001
        assert 0 <= p["algorithm.params.ent_coef"] < .1
        assert 1 <= p["algorithm.params.n_epochs"] < 4
        assert p["algorithm.params.batch_size"] in {64, 128}
    spec["sweep_seed"] += 1
    assert a != sweeps.generate_plan(spec, tune)
    spec["search"] = {"method": "random", "num_samples": 2, "parameters": {
        "algorithm.params.n_epochs": {"type": "choice", "values": [1]}}}
    with pytest.raises(ValueError, match="distinct samples"):
        sweeps.generate_plan(spec, tune)
    spec["allow_duplicates"] = True
    assert len(sweeps.generate_plan(spec, tune)) == 2


@pytest.mark.parametrize("updates,match", [
    ({"seeds": [42, 42]}, "Duplicate training seeds"),
    ({"resources_per_trial": {"cpu": 6, "gpu": 2}}, "multi-GPU"),
    ({"fixed_overrides": ["algorithm.params.learning_rate=0.3"]}, "Conflicting"),
    ({"fixed_overrides": ["experiment.total_steps=3"]}, "Conflicting"),
    ({"fixed_overrides": ["evaluation.seed=99"]}, "Conflicting"),
    ({"search": {"method": "random", "num_samples": 1, "parameters": {
        "algorithm.params.learning_rate": {"type": "loguniform", "low": 0, "high": 1}}}}, "positive"),
])
def test_invalid_specs_fail_before_ray(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        sweeps.read_spec(spec_file(tmp_path, **updates), adapter.REPO)


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_preflight_freezes_native_config_and_rejects_bad_paths_types_and_values(tmp_path, tune, algorithm):
    from core.scenario import load_and_expand_scenario
    from test_ray_tune_trials import settings
    spec = sweeps.read_spec(spec_file(tmp_path, seeds=[42, 43], algorithm=algorithm,
                                   scenario=adapter.SCENARIOS[algorithm]), adapter.REPO)
    spec["search"]["parameters"]["network.activation"] = {"type": "choice", "values": ["tanh"]}
    cfg = settings(tmp_path)
    cfg.update(sweep=spec, mode=algorithm, total_steps=spec["total_steps"], overrides=spec["fixed_overrides"],
               trial_plan=sweeps.generate_plan(spec, tune), scenarios={algorithm: spec["scenario"]})
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    for entry in cfg["trial_plan"]:
        config = load_and_expand_scenario(entry["input_path"])
        assert config["experiment"]["seed"] == entry["seed"]
        assert config["experiment"]["total_steps"] == 16384
        assert config["training_defaults"]["learning_rate"] == entry["parameters"]["algorithm.params.learning_rate"]
        assert config["training_defaults"]["batch_size"] == 128
        assert entry["schedule"]["lr_schedule"] == "linear"
        wb = config["wandb"]["tune"]
        assert wb["search_method"] == "grid" and wb["sweep_seed"] == 2026
        assert json.loads(wb["sampled_parameters"]) == entry["parameters"]
        from application.cli import parse_args, apply_cli_overrides
        effective = sweeps.trial_settings(cfg, entry)
        identity = {"algorithm": algorithm, "trial_id": "actual", "run_id": "actual", "wandb_id": "actual"}
        args = parse_args(adapter.trainer_command(effective, identity, tmp_path)[2:])
        launched = apply_cli_overrides(load_and_expand_scenario(args.scenario, overrides=args.parameter_overrides), args)
        assert launched["experiment"]["seed"] == entry["seed"]
        assert launched["training_defaults"]["learning_rate"] == entry["parameters"]["algorithm.params.learning_rate"]
        assert launched["wandb"]["id"] == "actual"
        assert launched["network"]["activation"] == "tanh"
    assert cfg["evaluation_protocol"]["selection"]["seed"] == 10042
    assert cfg["evaluation_protocol"]["final"]["seed"] == 20042
    for path, domain, match in [
        ("algorithm.params.bogus", {"type": "choice", "values": [1]}, "Unknown configuration path"),
        ("algorithm.params.n_epochs", {"type": "choice", "values": [1.5]}, "requires int"),
        ("algorithm.params.batch_size", {"type": "int", "low": 0, "high": 3}, "batch_size"),
    ]:
        bad = deepcopy(cfg)
        bad["overrides"] = []
        bad["sweep"]["search"] = {"method": "random", "num_samples": 1, "parameters": {path: domain}}
        bad["trial_plan"] = sweeps.generate_plan(bad["sweep"], tune)
        with pytest.raises(ValueError, match=match):
            sweeps.preflight(bad, adapter.trainer_command, adapter.REPO)


def test_fifo_plan_concurrency_and_restore_without_resampling(tmp_path, tune, monkeypatch):
    captured, restores = [], []
    result_configs = []

    class Tuner:
        def __init__(self, trainable, **kwargs):
            captured.append(kwargs)
            result_configs.extend({"algorithm": "ppo", "sweep_trial": entry} for entry in kwargs["param_space"]["sweep_trial"]["grid_search"])

        def fit(self):
            return [SimpleNamespace(config=config, metrics={"algorithm": "ppo", "status": "completed"},
                                    path="test", error=None) for config in result_configs]

        can_restore = staticmethod(lambda path: True)

        @classmethod
        def restore(cls, path, **kwargs):
            restores.append(kwargs)
            return object.__new__(cls)

    tune.Tuner = Tuner
    tune.with_resources = lambda fn, resources: (fn, resources)
    for name in ("TuneConfig", "RunConfig", "FailureConfig", "CheckpointConfig", "CLIReporter"):
        setattr(tune, name, lambda **kwargs: kwargs)
    ray = SimpleNamespace(tune=tune, init=lambda **kw: None, shutdown=lambda: None,
                          cluster_resources=lambda: {"GPU": 2}, nodes=lambda: [{"Alive": True, "Resources": {"CPU": 8, "GPU": 1}}] * 2)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv("RAY_ADDRESS", "test")
    monkeypatch.setattr(adapter, "current_revision", lambda: "revision")
    path = spec_file(tmp_path)
    root = tmp_path / "experiment"
    command = ["adapter", "--sweep-spec", str(path), "--root", str(root)]
    monkeypatch.setattr(sys, "argv", command)
    adapter.main()
    assert len(result_configs) == 3  # two GPUs, third trial remains in the plan
    assert captured[0]["tune_config"]["max_concurrent_trials"] == 2
    assert "scheduler" not in captured[0]["tune_config"]  # FIFO, no early stopping
    saved = json.loads((root / "experiment.json").read_text())
    plan = deepcopy(saved["trial_plan"])
    monkeypatch.setattr(adapter, "generate_plan", lambda *a: pytest.fail("Restoration resampled the sweep"))
    monkeypatch.setattr(sys, "argv", [*command, "--restore"])
    adapter.main()
    assert json.loads((root / "experiment.json").read_text())["trial_plan"] == plan
    assert restores[0]["resume_unfinished"] and restores[0]["resume_errored"]
    assert not restores[0]["restart_errored"] and "param_space" not in restores[0]
    monkeypatch.setattr(adapter, "current_revision", lambda: "unexpected")
    with pytest.raises(ValueError, match="Git revision"):
        adapter.main()
    monkeypatch.setattr(adapter, "current_revision", lambda: "revision")
    changed = deepcopy(saved["sweep"])
    changed["total_steps"] += 1
    with pytest.raises(ValueError, match="Sweep specification differs"):
        adapter.validate_restoration(saved, changed, saved["experiment_id"], root)
    saved["trial_plan"][0]["seed"] = 9
    with pytest.raises(ValueError, match="modified"):
        adapter.validate_restoration(saved, None, saved["experiment_id"], root)
    saved = json.loads((root / "experiment.json").read_text())
    Path(saved["trial_plan"][0]["input_path"]).write_text("modified frozen config")
    with pytest.raises(ValueError, match="Frozen sweep configuration"):
        adapter.validate_restoration(saved, None, saved["experiment_id"], root)


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_ranking_uses_native_best_and_handles_missing_and_nonfinite_results(tmp_path, algorithm):
    from test_torchrl_ppo import agent
    from training.hooks import EvaluationCheckpointHook as Hook
    from utils.torch_io import atomic_save
    from agents.common.checkpoints import read_checkpoint

    results = []
    for index, completion in enumerate((.9, 1., 1.)):
        out = tmp_path / str(index)
        out.mkdir()
        summary = ({"completion_rate": completion, "collision_rate": .1, "mean_net_progress": .9,
                    "mean_finish_steps": 20 - index} if algorithm == "ppo" else
                   {"team_both_finished_rate": completion, "team_completion_rate": 1., "team_collision_rate": .1,
                    "mean_net_progress": .9, "mean_clean_finish_time_s": 20 - index})
        strategy = "completion_progress" if algorithm == "ppo" else "team_completion"
        score = Hook.selection_score(summary, strategy)
        record = {**summary, "environment_steps": 100, "policy_version": 2, "selection_strategy": strategy,
                  "selection_score": list(score), "is_best": True, "episodes": 1, "episode_results": [{}],
                  "evaluation_protocol": {"name": "selection", "seeds": [10042], "max_steps": 100, "target_laps": 1}}
        (out / "evaluation_history.jsonl").write_text(json.dumps(record) + "\n")
        configuration = {"experiment": {"seed": 42, "total_steps": 100}, "algorithm": {"params": {"learning_rate": .001}}}
        entry = {"seed": 42, "parameters": {"algorithm.params.learning_rate": .001}}
        identity = {"run_id": str(index), "settings": {"sweep_trial": entry, "total_steps": 100,
                    "evaluation_protocol": {"selection": {"seed": 10042, "episodes": 1, "max_steps": 100, "target_laps": 1}}}}
        (out / "ray_trial.json").write_text(json.dumps(identity))
        agent().save_checkpoint(out / "best.pt", provenance={"run_id": str(index)}, configuration=configuration, selection=record)
        payload = read_checkpoint(out / "best.pt")
        payload["metadata"]["algorithm"] = algorithm
        atomic_save(payload, out / "best.pt")
        results.append({"status": "completed", "algorithm": algorithm, "training_output_dir": str(out),
                        "seed": 42, "hyperparameters": entry["parameters"], "error": None})
    missing = {**results[0], "training_output_dir": str(tmp_path / "missing")}
    # Discarded/replayed history may contain a better record than the actual
    # selected checkpoint. The checkpoint's own selection remains authoritative.
    with (tmp_path / "0/evaluation_history.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    ranked = sweeps.ranking([*results, missing, {**results[0], "status": "running"}])
    assert [Path(r["training_output_dir"]).name for r in ranked["ranked_trials"]] == ["2", "1", "0"]
    assert all(Path(r["best_checkpoint"]).parent == Path(r["training_output_dir"]) for r in ranked["ranked_trials"])
    assert len(ranked["diagnostics"]) == 2
    # Native unavailable tie-breakers stay -inf for comparison and null in JSON.
    path = tmp_path / "1/best.pt"
    payload = read_checkpoint(path)
    selection = payload["metadata"]["selection"]
    selection["mean_finish_steps" if algorithm == "ppo" else "mean_clean_finish_time_s"] = None
    selection["selection_score"] = [v if v != float("-inf") else None for v in Hook.selection_score(selection, strategy)]
    atomic_save(payload, path)
    with (tmp_path / "1/evaluation_history.jsonl").open("a") as stream:
        stream.write(json.dumps(selection) + "\n")
    ranked = sweeps.ranking(results)
    assert [Path(r["training_output_dir"]).name for r in ranked["ranked_trials"]] == ["2", "1", "0"]
    assert ranked["ranked_trials"][1]["unavailable_tie_breaker"]
    best = tmp_path / "0/best.pt"
    payload = read_checkpoint(best)
    payload["metadata"]["selection"]["selection_score"][0] = float("nan")
    atomic_save(payload, best)
    assert sweeps.ranking([results[0]])["diagnostics"][0]["ranking_status"] == "unranked"
    payload = read_checkpoint(path)
    payload["metadata"]["selection"]["episode_results"] = []
    atomic_save(payload, path)
    with (tmp_path / "1/evaluation_history.jsonl").open("a") as stream:
        stream.write(json.dumps(payload["metadata"]["selection"]) + "\n")
    assert "incomplete evaluation" in sweeps.ranking([results[1]])["diagnostics"][0]["ranking_reason"]


@pytest.mark.parametrize("dirty,untracked", [(1, ""), (0, "scripts/new.py")])
def test_revision_guard_rejects_unrecorded_source(monkeypatch, dirty, untracked):
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=dirty))
    monkeypatch.setattr(adapter.subprocess, "check_output", lambda *a, **kw: untracked)
    with pytest.raises(ValueError, match="Uncommitted experiment source"):
        adapter.current_revision()
