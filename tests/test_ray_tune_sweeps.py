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

    # Different selection horizons get separate ranks, never a global ranking.
    import shutil
    other = tmp_path / "other-protocol"
    shutil.copytree(tmp_path / "2", other)
    payload = read_checkpoint(other / "best.pt")
    record = payload["metadata"]["selection"]
    record["evaluation_protocol"]["max_steps"] = 101
    atomic_save(payload, other / "best.pt")
    (other / "evaluation_history.jsonl").write_text(json.dumps(record) + "\n")
    identity = json.loads((other / "ray_trial.json").read_text())
    identity["settings"]["evaluation_protocol"]["selection"]["max_steps"] = 101
    (other / "ray_trial.json").write_text(json.dumps(identity))
    separate = sweeps.ranking([results[2], {**results[2], "training_output_dir": str(other)}])
    assert [r["rank"] for r in separate["ranked_trials"]] == [1, 1]
    assert len(separate["comparison_groups"]) == 2
    record["evaluation_protocol"]["name"] = "final"
    atomic_save(payload, other / "best.pt")
    (other / "evaluation_history.jsonl").write_text(json.dumps(record) + "\n")
    assert not sweeps.ranking([{**results[2], "training_output_dir": str(other)}])["ranked_trials"]


@pytest.mark.parametrize("dirty,untracked", [(1, ""), (0, "scripts/new.py")])
def test_revision_guard_rejects_unrecorded_source(monkeypatch, dirty, untracked):
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=dirty))
    monkeypatch.setattr(adapter.subprocess, "check_output", lambda *a, **kw: untracked)
    with pytest.raises(ValueError, match="Uncommitted experiment source"):
        adapter.current_revision()


def architecture_settings(tmp_path, tune, *, parameters=None, algorithm="ppo", fixed=(), **updates):
    from test_ray_tune_trials import settings
    spec = yaml.safe_load((adapter.REPO / "scripts/sweeps/architectures_cpu_smoke.yaml").read_text())
    spec.update(algorithm=algorithm, scenario=f"scenarios/{algorithm}_serial_profile.yaml", **updates)
    if parameters is not None:
        spec["search"]["parameters"] = parameters
    spec["fixed_overrides"] += list(fixed)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    spec = sweeps.read_spec(path, adapter.REPO)
    cfg = settings(tmp_path, "disabled")
    cfg.update(sweep=spec, mode=algorithm, total_steps=spec["total_steps"], overrides=spec["fixed_overrides"],
               trial_plan=sweeps.generate_plan(spec, tune), scenarios={algorithm: spec["scenario"]},
               evaluation=True, checkpoint_every_steps=4, resources_per_trial=spec["resources_per_trial"])
    return cfg


def choice(values):
    return {"type": "choice", "values": values}


@pytest.mark.parametrize("preset", sorted(sweeps.NETWORKS))
def test_architecture_presets_freeze_reconstruct_and_capture_accounting(tmp_path, tune, preset):
    from core.scenario import load_and_expand_scenario
    from application.cli import parse_args, apply_cli_overrides
    cfg = architecture_settings(tmp_path, tune, parameters={"network": choice([preset])})
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    entry, = cfg["trial_plan"]
    assert entry["network_preset"] == preset
    assert entry["adaptation"] == "scratch"
    model = entry["model"]
    counts = model["parameter_counts"]
    assert counts["actor_total"] == counts["actor_trainable"] + counts["actor_frozen"]
    assert counts["learner_trainable"] == counts["actor_trainable"] + counts["critic_trainable"]
    assert model["encoder_type"] == ({"cnn1d_base": "cnn1d_fusion", "rescnn1d_base": "rescnn1d_fusion"}.get(preset, "identity"))
    args = parse_args(adapter.trainer_command(sweeps.trial_settings(cfg, entry),
        {"algorithm": "ppo", "trial_id": "ray", "run_id": "run", "wandb_id": "wb"}, tmp_path)[2:])
    frozen = apply_cli_overrides(load_and_expand_scenario(args.scenario, overrides=args.parameter_overrides), args)
    assert sweeps.configuration_digest(frozen) == entry["config_digest"]
    assert frozen["wandb"]["id"] == "wb"
    assert frozen["wandb"]["group"].endswith("pretraining")
    assert frozen["wandb"]["tune"]["parameter_counts"] == counts
    assert frozen["wandb"]["tune"]["resolved_network"] == model["resolved_network"]


def test_architecture_and_list_choices_are_deterministic_and_effective(tmp_path, tune):
    from core.scenario import load_and_expand_scenario
    cfg = architecture_settings(tmp_path, tune, parameters={
        "network": choice(["mlp_base", "mlp_wide"]),
        "network.critic_hidden_dims": choice([[16], [32, 16]])}, seeds=[4, 5])
    assert cfg["trial_plan"] == sweeps.generate_plan(cfg["sweep"], tune)
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    assert len({e["trial_key"] for e in cfg["trial_plan"]}) == 8
    for entry in cfg["trial_plan"]:
        config = load_and_expand_scenario(entry["input_path"])
        assert config["network"]["critic_hidden_dims"] == entry["parameters"]["network.critic_hidden_dims"]
    first = cfg["trial_plan"][0]
    snapshot = Path(first["input_path"]).read_bytes()
    (tmp_path / "experiment.json").write_text("valid existing manifest")
    with pytest.raises(ValueError, match="existing experiment"):
        sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    assert Path(first["input_path"]).read_bytes() == snapshot


@pytest.mark.parametrize("parameters,fixed,match", [
    ({"network": choice(["mlp_base", "cnn1d_base"]), "network.actor.encoder.channels": choice([[8, 16]])}, (), "Unknown configuration path"),
    ({"network": choice(["resmlp_base"]), "network.actor_hidden_dims": choice([[16]])}, (), "ineffective"),
    ({"network": choice(["mlp", "mlp_base"])}, (), "Duplicate effective"),
    ({"network": choice(["mlp_base"])}, ("network=mlp_wide",), "Conflicting"),
    ({"network": choice(["mlp_base"]), "adaptation": choice(["lora"])}, (), "Missing verified PPO"),
])
def test_invalid_architecture_combinations_and_shadowed_paths(tmp_path, tune, parameters, fixed, match):
    with pytest.raises(ValueError, match=match):
        cfg = architecture_settings(tmp_path, tune, parameters=parameters, fixed=fixed)
        sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)


def test_random_preflight_validates_unselected_categorical_interactions(tmp_path, tune):
    cfg = architecture_settings(tmp_path, tune, parameters={
        "network": choice(["mlp_base", "resmlp_base"]), "network.actor_hidden_dims": choice([[16]])})
    cfg["sweep"]["search"].update(method="random", num_samples=1)
    cfg["trial_plan"] = [{**cfg["trial_plan"][0], "parameters": {"network": "mlp_base", "network.actor_hidden_dims": [16]}}]
    with pytest.raises(ValueError, match="ineffective"):
        sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)


def make_sources(tmp_path, presets, seeds=(7,)):
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task
    from training.algorithms import learner_params, create_learner
    from ray_tune_sources import source_record
    records = []
    for preset in presets:
        for seed in seeds:
            config = load_and_expand_scenario("scenarios/ppo_serial_profile.yaml", overrides=[
                f"network={preset}", "execution.params.device=cpu", f"experiment.seed={seed}"])
            task = create_race_task(config)
            try:
                learner = create_learner("ppo", task.spec, learner_params(config, task.spec, "ppo"), training=False)
                checkpoint = tmp_path / f"{preset}-{seed}-best.pt"
                selection = {"selection_strategy": "completion_progress", "evaluation_protocol": {"name": "selection"}, "is_best": True}
                learner.save_checkpoint(checkpoint, configuration=config, selection=selection, provenance={"git": {"commit": "source-revision"}})
                records.append(source_record(checkpoint, preset, seed))
            finally:
                task.close()
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(yaml.safe_dump({"schema_version": 1, "sources": records}))
    return manifest, records


@pytest.mark.parametrize("preset", ["mlp_base", "resmlp_base", "cnn1d_base", "rescnn1d_base"])
def test_three_adaptation_arms_share_verified_source_and_protocol(tmp_path, tune, preset):
    manifest, records = make_sources(tmp_path, [preset])
    cfg = architecture_settings(tmp_path, tune, algorithm="mappo", parameters={
        "network": choice([preset]), "adaptation": choice(["scratch", "full_finetune", "lora"])},
        source_manifest=str(manifest), pretraining_seeds=[7], seeds=[4, 5])
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    assert len(cfg["trial_plan"]) == 6
    assert len({e["trial_key"] for e in cfg["trial_plan"]}) == 6
    for entry in cfg["trial_plan"]:
        if entry["adaptation"] == "scratch":
            assert not entry.get("source") and entry["pretraining_seed"] is None
        else:
            assert entry["source"] == records[0]
            assert entry["pretraining_seed"] == 7
        assert entry["model"]["actor_sharing"] == ("shared" if entry["adaptation"] == "lora" else "independent")
        counts = entry["model"]["parameter_counts"]
        assert bool(counts["lora_trainable"]) == (entry["adaptation"] == "lora")
        assert counts["exploration_trainable"] == 4
        assert bool(counts["neighbor_encoder_trainable"]) == ("cnn" in preset)
        if entry["adaptation"] == "lora":
            assert counts["actor_frozen"] > 0
            assert bool(counts["neighbor_encoder_trainable"]) == ("cnn" in preset)
    # No config/source identity is re-sampled or substituted on driver restore.
    cfg["manifest_digest"] = sweeps.digest(cfg)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter, "current_revision", lambda: cfg["git_sha"])
        adapter.validate_restoration(cfg, cfg["sweep"], cfg["experiment_id"], tmp_path)
        original = manifest.read_bytes()
        manifest.write_bytes(original + b"# changed manifest\n")
        with pytest.raises(ValueError, match="Source manifest changed"):
            adapter.validate_restoration(cfg, cfg["sweep"], cfg["experiment_id"], tmp_path)
        manifest.write_bytes(original)
        checkpoint = Path(records[0]["checkpoint"])
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="SHA-256 changed"):
            adapter.validate_restoration(cfg, cfg["sweep"], cfg["experiment_id"], tmp_path)


def test_manifest_missing_sources_are_actionable_and_seed_axes_independent(tmp_path, tune):
    manifest, records = make_sources(tmp_path, ["mlp_base"], seeds=(7, 8))
    cfg = architecture_settings(tmp_path, tune, algorithm="mappo", parameters={
        "network": choice(["mlp_base"]), "adaptation": choice(["scratch", "full_finetune", "lora"])},
        source_manifest=str(manifest), pretraining_seeds=[7, 8], seeds=[40, 41])
    assert len(cfg["trial_plan"]) == 10  # scratch once; two sources per transfer arm/seed
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    assert len({e["trial_key"] for e in cfg["trial_plan"]}) == 10
    manifest.write_text("schema_version: 1\nsources: []\n")
    cfg = architecture_settings(tmp_path, tune, algorithm="mappo", parameters={
        "network": choice(["mlp_base", "cnn1d_base"]), "adaptation": choice(["full_finetune", "lora"])},
        source_manifest=str(manifest), pretraining_seeds=[7])
    with pytest.raises(ValueError, match="Missing verified PPO sources") as error:
        sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    assert "('mlp_base', 7)" in str(error.value) and "('cnn1d_base', 7)" in str(error.value)
    from ray_tune_sources import verify_source
    Path(records[0]["checkpoint"]).unlink()
    with pytest.raises(ValueError, match="Missing PPO source"):
        verify_source(records[0])


@pytest.mark.parametrize("field,value,match", [("rank", 10000, "rank"), ("alpha", 0, "alpha")])
def test_lora_target_hyperparameters_validated_before_workers(tmp_path, tune, field, value, match):
    manifest, _ = make_sources(tmp_path, ["cnn1d_base"])
    cfg = architecture_settings(tmp_path, tune, algorithm="mappo", parameters={
        "network": choice(["cnn1d_base"]), f"adaptation.lora.{field}": choice([value])},
        fixed=["adaptation=lora"], source_manifest=str(manifest), pretraining_seeds=[7])
    with pytest.raises(Exception, match=match):
        sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)


@pytest.mark.parametrize("incompatibility", ["actor", "critic_state", "beams", "physics", "actions"])
def test_native_transfer_checks_complete_sources_and_destination_contracts(tmp_path, tune, incompatibility):
    from agents.common.checkpoints import read_checkpoint
    from utils.torch_io import atomic_save
    from ray_tune_sources import source_record
    manifest, records = make_sources(tmp_path, ["mlp_base"])
    source = Path(records[0]["checkpoint"])
    if incompatibility in {"critic_state", "actions"}:
        payload = read_checkpoint(source)
        if incompatibility == "critic_state":
            del payload["models"]["critic"][next(iter(payload["models"]["critic"]))]
        else:
            payload["metadata"]["contracts"]["actions"]["contract"]["speed_control"] = "direct"
        atomic_save(payload, source)
        record = source_record(source, "mlp_base", 7)
        manifest.write_text(yaml.safe_dump({"schema_version": 1, "sources": [record]}))
    fixed = {"actor": ["network.actor_hidden_dims=[32]"], "beams": ["environment.lidar_beams=1080"],
             "physics": ["environment.action_repeat=2"]}.get(incompatibility, [])
    cfg = architecture_settings(tmp_path, tune, algorithm="mappo", parameters={
        "network": choice(["mlp_base"]), "adaptation": choice(["full_finetune", "lora"])},
        fixed=fixed, source_manifest=str(manifest), pretraining_seeds=[7])
    with pytest.raises(ValueError):
        sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    assert not (tmp_path / "sweep_inputs").exists()


@pytest.mark.parametrize("algorithm,preset,adaptation", [("ppo", "rescnn1d_base", "scratch"), ("mappo", "cnn1d_base", "lora")])
def test_native_cpu_architecture_trial_checkpoint_ranking_and_recovery(tmp_path, tune, monkeypatch, algorithm, preset, adaptation):
    """Real native subprocess/optimizer/checkpoints; only Ray transport is stubbed."""
    import shutil
    from contextlib import nullcontext
    from agents.common.checkpoints import read_checkpoint
    parameters = {"network": choice([preset])}
    updates = {}
    if algorithm == "mappo":
        manifest, sources = make_sources(tmp_path, [preset])
        updates.update(source_manifest=str(manifest), pretraining_seeds=[7])
        parameters["adaptation"] = choice([adaptation])
    cfg = architecture_settings(tmp_path, tune, algorithm=algorithm, parameters=parameters, **updates)
    cfg["trial_timeout"] = 120
    cfg["poll_interval"] = .05
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    saved_checkpoint, reports = None, []

    class Checkpoint:
        def __init__(self, directory):
            self.directory = Path(directory)

        @classmethod
        def from_directory(cls, directory):
            return cls(directory)

        def as_directory(self):
            return nullcontext(self.directory)

    def report(values, checkpoint=None):
        nonlocal saved_checkpoint
        reports.append(values)
        if checkpoint:
            destination = tmp_path / f"registered-{len(reports)}"
            shutil.copytree(checkpoint.directory, destination)
            saved_checkpoint = Checkpoint(destination)

    transport = SimpleNamespace(get_context=lambda: SimpleNamespace(get_trial_id=lambda: "native-smoke"),
                                get_checkpoint=lambda: saved_checkpoint, report=report, Checkpoint=Checkpoint)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=transport, get_gpu_ids=lambda: []))
    trial = {"algorithm": algorithm, "settings": cfg, "sweep_trial": cfg["trial_plan"][0]}
    adapter.trial(trial)
    assert reports[-1]["status"] == "completed" and reports[-1]["environment_steps"] >= 8
    out = Path(reports[-1]["training_output_dir"])
    before = read_checkpoint(out / "latest.pt")
    assert before["training"]["optimizer"]["state"]
    result = {**reports[-1], "error": None}
    ranked = sweeps.ranking([result])
    assert len(ranked["ranked_trials"]) == 1, ranked["diagnostics"]
    assert ranked["ranked_trials"][0]["network_preset"] == preset
    assert ranked["ranked_trials"][0]["parameter_counts"]["learner_trainable"] > 0
    if algorithm == "ppo":
        from ray_tune_sources import source_record, verify_source
        record = source_record(out / "best.pt", preset, 42)
        verify_source(record)
    identity = json.loads((out / "ray_trial.json").read_text())
    adapter.trial(trial)  # Restore the completed budget; never restart progress.
    after = read_checkpoint(out / "latest.pt")
    assert after["training"]["progress"] == before["training"]["progress"]
    assert json.loads((out / "ray_trial.json").read_text()) == identity
    assert after["metadata"]["network"] == before["metadata"]["network"]
    assert after["metadata"]["structure"] == before["metadata"]["structure"]
    assert after["metadata"]["adaptation"] == before["metadata"]["adaptation"]
    assert after["training"]["optimizer"]["param_groups"] == before["training"]["optimizer"]["param_groups"]
    for key, state in before["training"]["optimizer"]["state"].items():
        for slot, value in state.items():
            assert value.equal(after["training"]["optimizer"]["state"][key][slot])
    assert all(value.equal(after["models"]["actor"][key]) for key, value in before["models"]["actor"].items())
    altered = deepcopy(identity)
    altered["settings"]["sweep_trial"]["config_digest"] = "0" * 64
    with pytest.raises(ValueError, match="frozen architecture/adaptation"):
        adapter.checkpoint_payload(out / "latest.pt", altered)
    if algorithm == "mappo":
        import subprocess
        wrong = subprocess.run([sys.executable, str(adapter.REPO / "run.py"), "--scenario", str(out / "trial_input.yaml"),
            "--output-dir", str(tmp_path / "wrong-source"), "--run-id", "wrong-source", "--set",
            '++wandb.tune.source_sha256="' + "0" * 64 + '"'], cwd=adapter.REPO, text=True, capture_output=True, timeout=60)
        assert wrong.returncode != 0
        assert "Loaded PPO source differs from the frozen sweep SHA-256" in wrong.stderr + wrong.stdout
        assert not (tmp_path / "wrong-source/latest.pt").exists()
        source = Path(sources[0]["checkpoint"])
        source.write_bytes(source.read_bytes() + b"changed after preflight")
        with pytest.raises(ValueError, match="SHA-256 changed"):
            adapter.trial(trial)


@pytest.mark.parametrize("preset,parameters", [
    ("resmlp_base", {"network.actor.width": choice([16, 32]), "network.actor.num_blocks": choice([1]), "network.activation": choice(["tanh"])}),
    ("rescnn1d_base", {"network.actor.encoder.channels": choice([[4, 8], [8, 16]]),
                       "network.actor.encoder.num_blocks": choice([1]), "network.actor.encoder.lidar_features": choice([16])}),
    ("mlp_base", {"network.actor_hidden_dims": choice([[16], [32, 16]])}),
])
def test_architecture_specific_searches_use_separate_validated_spaces(tmp_path, tune, preset, parameters):
    cfg = architecture_settings(tmp_path, tune, parameters=parameters, fixed=[f"network={preset}"])
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    assert len(cfg["trial_plan"]) == 2
    assert len({e["config_digest"] for e in cfg["trial_plan"]}) == 2
    assert all(e["network_preset"] == preset for e in cfg["trial_plan"])


def test_example_counts_and_preflight_only_leave_existing_experiments_untouched(tmp_path, tune, monkeypatch, capsys):
    for filename, count in [("ppo_architectures.yaml", 12), ("mappo_architecture_adaptation.yaml", 36),
                            ("architectures_cpu_smoke.yaml", 4)]:
        spec = sweeps.read_spec(adapter.REPO / "scripts/sweeps" / filename, adapter.REPO)
        assert len(sweeps.generate_plan(spec, tune)) == count
    cfg = architecture_settings(tmp_path, tune, parameters={"network": choice(["mlp_small"])})
    existing = tmp_path / "experiment.json"
    existing.write_text("existing valid manifest")
    before = existing.read_bytes()
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=tune))
    monkeypatch.setattr(sys, "argv", ["adapter", "--sweep-spec", str(tmp_path / "spec.yaml"),
                                      "--preflight-only", "--root", str(tmp_path)])
    adapter.main()
    assert "PREFLIGHT PASSED" in capsys.readouterr().out
    assert existing.read_bytes() == before
    assert not (tmp_path / "sweep_inputs").exists()


def test_production_preflight_preserves_historical_training_and_fixed_1080_beams(tmp_path, tune):
    from test_ray_tune_trials import settings
    from core.scenario import load_and_expand_scenario
    spec = sweeps.read_spec(adapter.REPO / "scripts/sweeps/ppo_architectures.yaml", adapter.REPO)
    cfg = settings(tmp_path, "disabled")
    cfg.update(sweep=spec, mode="ppo", total_steps=spec["total_steps"], overrides=spec["fixed_overrides"],
               trial_plan=sweeps.generate_plan(spec, tune), scenarios={"ppo": spec["scenario"]}, evaluation=True,
               resources_per_trial=spec["resources_per_trial"])
    sweeps.preflight(cfg, adapter.trainer_command, adapter.REPO)
    historical = load_and_expand_scenario(spec["scenario"])
    assert historical["environment"]["lidar_beams"] == 108
    for entry in cfg["trial_plan"]:
        config = load_and_expand_scenario(entry["input_path"])
        assert config["environment"]["lidar_beams"] == 1080
        assert config["experiment"]["parallel_envs"] == 8
        assert config["experiment"]["envs_per_worker"] == 8
        for key in ("n_steps", "batch_size", "n_epochs", "learning_rate", "lr_schedule", "learning_rate_end", "frames_per_batch"):
            assert config["training_defaults"][key] == historical["training_defaults"][key]
    assert len({e["config_digest"] for e in cfg["trial_plan"]}) == 12


def test_default_scenario_adaptation_preserves_pretraining_seed_axis(tmp_path, tune):
    manifest, _ = make_sources(tmp_path, ["mlp_base"], seeds=(7, 8))
    cfg = architecture_settings(tmp_path, tune, algorithm="mappo", parameters={"network": choice(["mlp_base"])},
                                source_manifest=str(manifest), pretraining_seeds=[7, 8])
    spec = deepcopy(cfg["sweep"])
    spec["scenario"] = str(adapter.REPO / "scenarios/mappo_2v2_completion_lora.yaml")
    plan = sweeps.generate_plan(spec, tune)
    assert [e["pretraining_seed"] for e in plan] == [7, 8]
