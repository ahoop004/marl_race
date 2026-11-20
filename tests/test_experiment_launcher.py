"""Focused launch checks: real Hydra resolution, fake training, no Slurm submission."""
from argparse import Namespace
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

spec = importlib.util.spec_from_file_location("launch_experiment", Path(__file__).parents[1] / "scripts/launch_experiment.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def arguments(root, **kwargs):
    values = dict(root=root, scenario="scenarios/ppo_lora_families.yaml", source=None,
                  overrides=[], seed=None, profile=None, steps_per_task=None, workers=None,
                  envs_per_worker=None, reserved_cpus=None, resume=False, prepare_only=False)
    values.update(kwargs)
    return Namespace(**values)


@pytest.fixture(scope="module")
def family_configs(tmp_path_factory):
    return launch.prepare_configs(arguments(tmp_path_factory.mktemp("plan")), 16)[0]


@pytest.mark.parametrize("tasks", [1, 2, 4, 5, 7])
def test_rank_striding_executes_every_family_once(tasks, family_configs, tmp_path, monkeypatch):
    executions = {}
    monkeypatch.setattr(launch, "prepare_manifest", lambda *args: {"items": family_configs})
    monkeypatch.setattr(launch, "run_item", lambda args, name, config, identity:
                        executions.setdefault(identity[0], []).append(name))
    for rank in range(tasks):
        monkeypatch.setattr(launch, "task_identity", lambda: (rank, tasks, list(range(16))))
        launch.main(["--root", str(tmp_path), "--scenario", "unused.yaml"])
    assigned = [name for names in executions.values() for name in names]
    assert len(assigned) == len(set(assigned)) == len(family_configs) == 5
    if tasks == 4:
        assert executions[0] == ["straight", "s_curve"]
    if tasks == 5:
        assert all(len(names) == 1 for names in executions.values())


def test_affinity_workers_alignment_and_explicit_overrides(family_configs):
    original = family_configs["straight"]
    for cpus, expected in ((16, 12), (24, 16), (32, 24)):
        config = launch.resolve_collectors(deepcopy(original), cpus)
        assert config["experiment"]["parallel_envs"] == expected
        assert config["training_defaults"] == original["training_defaults"]
    # Batch-compatible algorithms use the full CPU-minus-reservation defaults.
    for cpus, expected in ((16, 12), (24, 20), (32, 28)):
        config = deepcopy(original)
        config["training_defaults"].update(frames_per_batch=1680, n_steps=3360, batch_size=112)
        assert launch.resolve_collectors(config, cpus)["experiment"]["parallel_envs"] == expected
    config = deepcopy(original)
    config["execution"]["collector"].update(workers=8, envs_per_worker=2)
    assert launch.resolve_collectors(config, 16)["experiment"]["parallel_envs"] == 8
    config["execution"]["collector"]["workers"] = 7
    with pytest.raises(ValueError, match="optimization targets"):
        launch.resolve_collectors(config, 16)
    with pytest.raises(ValueError, match="affinity has 8"):
        launch.resolve_collectors(deepcopy(original), 8, frozen=True)


def test_scenario_budget_and_continual_conditions(tmp_path):
    configs, _ = launch.prepare_configs(arguments(tmp_path, overrides=["experiment.episodes=19"]), 16)
    assert all(c["experiment"]["episodes"] == 19 for c in configs.values())
    assert all(not c["environment"].get("no_progress") for c in configs.values())
    for profile, steps in (("pilot", 2000000), ("full", 10000000)):
        configs, _ = launch.prepare_configs(arguments(tmp_path, scenario="scenarios/ppo_continual.yaml", profile=profile), 16)
        assert list(configs) == ["full_finetune", "ppo_lora"]
        assert {c["adaptation"]["mode"] for c in configs.values()} == {"full_finetune", "lora"}
        assert all(c["experiment"]["total_steps"] == steps * len(c["continual"]["families"]) for c in configs.values())


def test_freezing_recovery_completion_and_source_identity(tmp_path, monkeypatch):
    source = tmp_path / "base.pt"
    source.write_bytes(b"immutable source fixture")
    args = arguments(tmp_path / "run", source=source, prepare_only=True)
    data = launch.prepare_manifest(args, 16)
    config = data["items"]["straight"]
    assert data["git_commit"] and data["source"]["sha256"] == launch.sha256(source)
    assert (args.root / "source.pt").stat().st_mode & 0o222 == 0
    assert set(data["items"]) == {"straight", "sweeper", "hairpin", "chicane", "s_curve"}
    launch.run_item(args, "straight", config, (0, 4, list(range(16))))
    out = args.root / "straight"
    frozen = (out / "launch_config.yaml").read_bytes()
    with pytest.raises(ValueError, match="Existing experiment"):
        launch.prepare_manifest(args, 16)
    with pytest.raises(ValueError, match="Existing experiment"):
        launch.run_item(args, "straight", config, (0, 4, list(range(16))))
    args.resume, args.source, args.scenario = True, None, "no-longer-present.yaml"
    # No composition on recovery; larger CPU and different GPU counts retain settings.
    assert launch.prepare_manifest(args, 32) == data
    manifest = args.root / "launch_manifest.yaml"
    original_manifest = manifest.read_bytes()
    with monkeypatch.context() as changed_code:
        changed_code.setattr(launch, "source_code_sha256", lambda: "different training code")
        with pytest.raises(ValueError, match="revision compatibility cannot be proven"):
            launch.prepare_manifest(args, 32)
        args.allow_code_change = True
        assert launch.prepare_manifest(args, 32) == data
        args.allow_code_change = False
    assert manifest.read_bytes() == original_manifest
    monkeypatch.setattr(launch, "task_identity", lambda: (0, 1, list(range(32))))
    launch.main(["--root", str(args.root), "--resume", "--prepare-only"])
    queued = yaml.safe_load((args.root / "s_curve/launch_config.yaml").read_text())
    assert queued == data["items"]["s_curve"]
    with pytest.raises(ValueError, match="Frozen collector requires"):
        launch.run_item(args, "straight", config, (0, 2, list(range(12))))
    latest = out / "latest.pt"
    latest.write_bytes(b"checkpoint fixture")
    commands = []
    def train(command, **kwargs):
        commands.append(command)
        (out / "final.pt").write_bytes(b"final fixture")
        (out / "run_summary.json").write_text(json.dumps({"completed_episodes": config["experiment"]["episodes"]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launch, "validate_gpu", lambda config: None)
    monkeypatch.setattr(launch.subprocess, "run", train)
    args.prepare_only = False
    launch.run_item(args, "straight", config, (0, 2, list(range(32))))
    assert commands[0][-2:] == ["--resume", str(latest)]
    assert (out / "launch_config.yaml").read_bytes() == frozen
    assert yaml.safe_load(frozen)["wandb"]["id"] == config["wandb"]["id"]
    # run.py composes this same standalone snapshot, without consulting its old defaults.
    reloaded = launch.load_and_expand_scenario(out / "launch_config.yaml")
    for field in ("experiment", "algorithm", "training_defaults", "environment", "agents", "wandb"):
        assert reloaded[field] == config[field]
    launch.run_item(args, "straight", config, (0, 5, list(range(32))))
    assert len(commands) == 1  # Verified completed items never launch again.
    snapshot = args.root / "source.pt"
    snapshot.chmod(0o644)
    snapshot.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="content changed"):
        launch.prepare_manifest(args, 32)


def test_older_manifest_revision_guard_and_legacy_import(tmp_path, monkeypatch):
    # Exercise real Git checks against a clean fixture independent of worktree edits.
    checkout = tmp_path / "checkout"
    launch.subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(launch.REPO), str(checkout)],
                          check=True)
    launch.subprocess.run(["git", "restore", "--source=HEAD", "--staged", "--worktree", "--", "src", "run.py"],
                          cwd=checkout, check=True)
    commit = launch.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    with monkeypatch.context() as fixture:
        fixture.setattr(launch, "REPO", checkout)
        record = dict(git_commit=commit, git_dirty=False)
        launch.validate_revision(record)
        for incompatible in ({}, {**record, "git_dirty": True}, {**record, "legacy_import": True}):
            with pytest.raises(ValueError, match="allow-code-change"):
                launch.validate_revision(incompatible)
        (checkout / "run.py").write_text("# different training source\n")
        with pytest.raises(ValueError, match="allow-code-change"):
            launch.validate_revision(record)
    # Importing snapshots must not label today's source as the original revision.
    args = arguments(tmp_path, resume=True, scenario="scenarios/ppo_generated_lora_20k.yaml")
    with pytest.raises(ValueError, match="revision compatibility cannot be proven"):
        launch.prepare_manifest(args, 16)
    assert not (tmp_path / "launch_manifest.yaml").exists()
    config = {"adaptation": {"checkpoint": None}, "wandb": {"id": "original-id"}}
    out = tmp_path / "train"
    out.mkdir()
    (out / "launch_config.yaml").write_text(yaml.safe_dump(config))
    def legacy_template(path):
        assert path == "scenarios/ppo_lora_families.yaml"
        return {}
    monkeypatch.setattr(launch, "compose_configuration", legacy_template)
    args.allow_code_change = True
    data = launch.prepare_manifest(args, 16)
    assert data["git_commit"] is data["git_dirty"] is data["source_code_sha256"] is None
    assert data["items"]["train"] == config


def test_training_source_identity_includes_uncommitted_files(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "run.py").write_text("# entry point\n")
    source = tmp_path / "src/model.py"
    source.write_text("# original\n")
    monkeypatch.setattr(launch, "REPO", tmp_path)
    original = launch.source_code_sha256()
    source.write_text("# changed\n")
    assert launch.source_code_sha256() != original
    source.write_text("# original\n")
    assert launch.source_code_sha256() == original
    (tmp_path / "src/new.py").write_text("# untracked training code\n")
    assert launch.source_code_sha256() != original


def test_task_gpu_visibility_is_relative_and_preserved(family_configs, monkeypatch):
    import torch
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-task-local-uuid")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    launch.validate_gpu(family_configs["straight"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-task-local-uuid"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 5)
    with pytest.raises(ValueError, match="one visible GPU"):
        launch.validate_gpu(family_configs["straight"])


def test_cpu_preflight_assigns_one_family_per_node(tmp_path, monkeypatch, capsys):
    import torch
    from test_torchrl_ppo import agent

    root = tmp_path / "run"
    root.mkdir()
    (root / ".allocation.lock").write_text("shared\n")
    source = tmp_path / "original.pt"
    agent().save_checkpoint(source)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(launch.os, "sched_getaffinity", lambda pid: set(range(32)))
    for key, value in {"SLURM_NTASKS": "5", "SLURM_JOB_NUM_NODES": "5", "SLURM_LOCALID": "0",
                       "SLURM_CPUS_PER_TASK": "32", "SLURM_JOB_ID": "local-preflight"}.items():
        monkeypatch.setenv(key, value)
    command = ["--scenario", "scenarios/ppo_lora_families.yaml", "--root", str(root), "--source", str(source),
               "--workers", "24", "--reserved-cpus", "8", "--independent-cpu", "--verify-allocation",
               "--shared-token", "shared", "--set", "execution.params.device=cpu",
               "--set", "execution.params.collection_device=cpu", "--set", "evaluation.enabled=true",
               "--set", "experiment.async_evaluation=true"]
    for rank, family in enumerate(("straight", "sweeper", "hairpin", "chicane", "s_curve")):
        monkeypatch.setenv("SLURM_PROCID", str(rank))
        monkeypatch.setenv("SLURMD_NODENAME", f"simulated-{rank}")
        launch.main(command)
        output = capsys.readouterr().out
        assert f"Verified {family}: rank={rank} node=simulated-{rank} cpus=32 collectors=24x4" in output
        assert "eval_workers=1 cuda=0" in output
    manifest = yaml.safe_load((root / "launch_manifest.yaml").read_text())
    assert manifest["source"]["sha256"] == launch.sha256(source)
    assert all(config["experiment"]["episodes"] == 30000 for config in manifest["items"].values())
    assert len({config["_launch"]["run_id"] for config in manifest["items"].values()}) == 5
    assert all(set(config["maps"]["selection"]).isdisjoint(config["maps"]["final_test"])
               for config in manifest["items"].values())


def test_recovery_before_first_checkpoint_preserves_attempt_artifacts(tmp_path, family_configs, monkeypatch):
    args = arguments(tmp_path, prepare_only=True)
    config = deepcopy(family_configs["straight"])
    launch.run_item(args, "straight", config, (0, 1, list(range(16))))
    out = tmp_path / "straight"
    (out / "train.log").write_text("earlier attempt\n")
    (out / "episode_metrics.csv").write_text("historical rows\n")
    args.resume, args.prepare_only = True, False
    monkeypatch.setattr(launch, "validate_gpu", lambda config: None)
    commands = []
    def train(command, **kwargs):
        commands.append(command)
        (out / "final.pt").write_bytes(b"final fixture")
        (out / "run_summary.json").write_text(json.dumps({"completed_episodes": config["experiment"]["episodes"]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launch.subprocess, "run", train)
    launch.run_item(args, "straight", config, (0, 1, list(range(16))))
    assert "--resume" not in commands[0]
    assert "earlier attempt" in (out / "train.log").read_text()
    assert next(out.glob("uncheckpointed_*/episode_metrics.csv")).read_text() == "historical rows\n"


def test_slurm_runtime_inherits_allocation_and_propagates_failure(tmp_path):
    import subprocess
    bindir = tmp_path / "bin"
    bindir.mkdir()
    srun = bindir / "srun"
    srun.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                    "open(os.environ['LAUNCH_ARGUMENTS'], 'w').write(json.dumps(sys.argv[1:]))\n"
                    "sys.exit(int(os.environ.get('LAUNCH_EXIT', '0')))\n")
    srun.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", CRUN="/bin/true",
               REPO_DIR=str(launch.REPO), RUN_ROOT=str(tmp_path / "run"), SLURM_JOB_ID="test",
               SLURM_NTASKS="5", SLURM_CPUS_PER_TASK="32", LAUNCH_ARGUMENTS=str(tmp_path / "args.json"))
    script = launch.REPO / "scripts/hpc/experiment.sbatch"
    assert "#SBATCH --mem=" not in script.read_text()
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = json.loads((tmp_path / "args.json").read_text())
    assert "--gpu-bind=single:1" in command and "--cpu-bind=cores" in command
    assert not any(arg.startswith(("--ntasks", "--cpus-per-task", "--gpus-per-task", "--nodes")) for arg in command)
    result = subprocess.run(["bash", str(script)], env=dict(env, LAUNCH_EXIT="7"), capture_output=True, text=True)
    assert result.returncode == 7 and "RESUME=1" in result.stderr

    scontrol = bindir / "scontrol"
    scontrol.symlink_to("/bin/true")
    cpu_script = launch.REPO / "scripts/hpc/experiment_cpu.sbatch"
    assert "#SBATCH --partition=main" in cpu_script.read_text()
    assert "--gpus" not in cpu_script.read_text()
    result = subprocess.run(["bash", str(cpu_script)], env=dict(env, SLURM_JOB_NUM_NODES="5",
                            SLURM_JOB_PARTITION="main", PREFLIGHT_ONLY="1"), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = json.loads((tmp_path / "args.json").read_text())
    assert "--independent-cpu" in command and "--verify-allocation" in command
    assert "--ntasks=5" in command and "--ntasks-per-node=1" in command and "--cpus-per-task=32" in command
    assert not any(arg.startswith(("--gpu-bind", "--gpus", "--gres")) for arg in command)
