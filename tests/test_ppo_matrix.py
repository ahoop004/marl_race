"""Focused profiling preflight and submission safeguards; no Slurm jobs or GPUs."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time
from types import SimpleNamespace

import pytest
import yaml

from test_ray_tune_trials import adapter
from test_ray_tune_sweeps import tune  # shared mocked Ray domains; native Hydra validation is real
from core.scenario import load_and_expand_scenario, ScenarioError
from application.configuration import validate_experiment_scenario


@pytest.fixture
def matrix_env(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    env = os.environ.copy()
    env.update(PATH=str(bindir) + os.pathsep + env["PATH"], CRUN=str(bindir / "crun"),
               VENV=str(tmp_path), PROFILE_MATRIX_ROOT=str(tmp_path / "matrices"),
               MATRIX_TEST_EVENTS=str(tmp_path / "events"))
    scripts = {
        "crun": '''#!/usr/bin/env python3
import os,sys,pathlib
args=sys.argv
spec=pathlib.Path(args[args.index('--sweep-spec')+1])
with open(os.environ['MATRIX_TEST_EVENTS'],'a') as f: f.write('validate:'+spec.stem+'\\n')
if spec.stem==os.environ.get('FAIL_PROFILE'):
 print('ValueError: Unknown configuration path: experiment.measure_bogus',file=sys.stderr)
 sys.exit(1)
print('Validation entrypoint exercised: '+spec.stem)
''',
        "git": '''#!/usr/bin/env bash
if [[ "$1" == branch ]]; then echo multiNode; fi
''',
        "sbatch": '''#!/usr/bin/env python3
import os,sys
p=os.environ['MATRIX_TEST_EVENTS']
with open(p) as f: n=sum(line.startswith('submit:') for line in f)+1
with open(p,'a') as f: f.write('submit:'+' '.join(sys.argv[1:])+'\\n')
if str(n)==os.environ.get('FAIL_SBATCH_AT'): sys.exit(1)
print(8000+n)
''',
    }
    for name, source in scripts.items():
        path = bindir / name
        path.write_text(source)
        path.chmod(0o755)
    return env


def submitter(env, *args):
    return subprocess.run(["bash", "scripts/hpc/submit_ppo_matrix.sh", *args], cwd=adapter.REPO,
                          env=env, text=True, capture_output=True, timeout=30)


def test_pipeline_plan_separates_budgets_and_submits_held_dependency_chain(tmp_path, tune, monkeypatch):
    import ppo_mappo_pipeline as pipeline
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=tune))
    root = tmp_path / "pipeline"
    value = pipeline.write_plan(root, "pipeline-test", 120000000, "24:00:00", 3, "revision")
    ppo = yaml.safe_load(Path(value["stages"]["ppo"]["spec"]).read_text())
    mappo = yaml.safe_load(Path(value["stages"]["mappo"]["spec"]).read_text())
    assert ppo["total_steps"] == 120000000 and mappo["total_steps"] == 24576000
    assert ppo["seeds"] == [42] and mappo["seeds"] == [42, 43]
    assert mappo["resources_per_trial"] == {"cpu": 10, "gpu": 1}
    assert mappo["max_concurrent_trials"] == 4
    assert "environment.max_steps=8000" in mappo["fixed_overrides"]
    pipeline.preflight_stage(value, "ppo")
    # Validate all six real destination configurations using a native, untrained
    # model fixture. No collectors, environment steps or Ray cluster are started.
    from core.task_builder import create_race_task
    from training.algorithms import learner_params, create_learner
    from ray_tune_sources import source_record
    source_dir = root / "sources"
    source_dir.mkdir()
    source_config = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml",
        overrides=["network=mlp_base", "environment.lidar_beams=108", "execution.params.device=cpu"])
    task = create_race_task(source_config)
    try:
        policy = create_learner("ppo", task.spec, learner_params(source_config, task.spec, "ppo"), training=False)
        checkpoint = source_dir / "best.pt"
        policy.save_checkpoint(checkpoint, configuration=source_config, selection={
            "selection_strategy": "completion_progress", "evaluation_protocol": {"name": "selection"}, "is_best": True})
        record = source_record(checkpoint, "mlp_base", 42)
    finally:
        task.close()
    (source_dir / "manifest.yaml").write_text(yaml.safe_dump({"schema_version": 1, "sources": [record]}))
    pipeline.preflight_stage(value, "mappo")
    submissions, releases = [], []
    def submit(command, **kwargs):
        submissions.append(command)
        return str(8000 + len(submissions)) + "\n"
    monkeypatch.setattr(pipeline.subprocess, "check_output", submit)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: releases.append(command))
    pipeline.queue_pipeline(value)
    assert len(submissions) == 4
    assert all("--hold" in c and "--signal=B:USR1@240" in c for c in submissions)
    assert "--nodes=1" in submissions[0] and "--gres=gpu:1" in submissions[0]
    assert "--nodes=2" in submissions[1] and "--ntasks=2" in submissions[1] and "--gres=gpu:2" in submissions[1]
    assert [next((a for a in c if a.startswith("--dependency")), None) for c in submissions] == [
        None, "--dependency=afterok:8001", "--dependency=afterok:8002", "--dependency=afterok:8003"]
    assert releases == [["scontrol", "release", str(job)] for job in range(8001, 8005)]
    assert json.loads((root / "jobs.json").read_text())["status"] == "released"
    with pytest.raises(ValueError, match="already submitted"):
        pipeline.queue_pipeline(value)
    Path(value["stages"]["ppo"]["spec"]).chmod(0o644)
    Path(value["stages"]["ppo"]["spec"]).write_text("modified: true")
    monkeypatch.setattr(pipeline, "current_revision", lambda: "revision")
    with pytest.raises(ValueError, match="modified"):
        pipeline.load_pipeline(root)


@pytest.mark.parametrize("restart,expected", [(0, ["evaluating", "interrupted", "requeue"]),
                                              (1, ["evaluating", "interrupted", "failed"])])
def test_pipeline_pretimeout_requeues_same_job_with_explicit_limit(tmp_path, restart, expected):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    events = tmp_path / "events"
    commands = {
        "git": '#!/usr/bin/env bash\nif [[ "$1" == branch ]]; then echo multiNode; fi\n',
        "scontrol": '#!/usr/bin/env bash\nprintf "%s\\n" "$1" >> "$TEST_EVENTS"\n',
        "crun": '''#!/usr/bin/env python3
import os,sys,time
from pathlib import Path
action=sys.argv[sys.argv.index('--stage-action')+1]
if action=='environment':
 print('export PIPELINE_MAX_RESTARTS=1')
else:
 with Path(os.environ['TEST_EVENTS']).open('a') as f:
  f.write(('evaluating' if action=='evaluate' else action)+'\\n')
 if action=='evaluate': time.sleep(60)
'''}
    for name, source in commands.items():
        path = bindir / name
        path.write_text(source)
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "CRUN": str(bindir / "crun"), "VENV": str(tmp_path), "RAY_TUNE_REPO": str(adapter.REPO),
        "PIPELINE_ROOT": str(tmp_path), "PIPELINE_STAGE": "team", "SLURM_JOB_ID": "9001",
        "SLURM_RESTART_COUNT": str(restart), "TEST_EVENTS": str(events)}
    process = subprocess.Popen(["bash", str(adapter.REPO / "scripts/hpc/ray_tune_matrix.sbatch")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not events.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert events.exists()
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 75, stdout + stderr
        assert events.read_text().splitlines() == expected
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_all_seven_profiles_pass_native_preflight_without_cluster_or_manifest(tmp_path, matrix_env, tune, monkeypatch):
    cp = submitter(matrix_env, "--dry-run", "--tag", "validated")
    assert cp.returncode == 0, cp.stderr
    events = Path(matrix_env["MATRIX_TEST_EVENTS"]).read_text().splitlines()
    assert len(events) == 7 and not any(e.startswith("submit:") for e in events)
    manifest = Path(matrix_env["PROFILE_MATRIX_ROOT"]) / "validated/jobs.tsv"
    with manifest.open() as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    expected = [("w04_n1g1", 1, 1, 4), ("w08_n1g1", 1, 1, 8), ("w16_n1g1", 1, 1, 16),
                ("n02_g1", 2, 1, 16), ("n03_g1", 3, 1, 16), ("n02_g2", 2, 2, 8), ("n03_g2", 3, 2, 8)]
    assert [(r["profile"], int(r["nodes"]), int(r["gpus_per_node"]), int(r["workers"])) for r in records] == expected
    # A real native preflight of all 42 trials, using the same CLI mode as crun.
    # Only Ray's parameter generator is mocked because Ray is not installed here.
    def forbidden(*args, **kwargs):
        pytest.fail("Validation attempted cluster startup, training, or production revision handling")
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=tune, init=forbidden))
    monkeypatch.setattr(adapter, "current_revision", forbidden)
    monkeypatch.setattr(adapter, "trial", forbidden)
    production = tmp_path / "production"
    production.mkdir()
    sentinel = production / "experiment.json"
    sentinel.write_text("keep production manifest")
    for record in records:
        spec = manifest.parent / "specs" / (record["profile"] + ".yaml")
        raw = yaml.safe_load(spec.read_text())
        assert raw["seeds"] == [42, 43, 44, 45, 46, 47]
        assert raw["total_steps"] == 262144 and raw["resources_per_trial"]["gpu"] == 1
        assert "algorithm.params.n_steps=32768" in raw["fixed_overrides"]
        assert "algorithm.params.frames_per_batch=4096" in raw["fixed_overrides"]
        assert "algorithm.params.batch_size=1024" in raw["fixed_overrides"]
        assert "algorithm.params.n_epochs=10" in raw["fixed_overrides"]
        monkeypatch.setattr(sys, "argv", ["adapter", "--preflight-only", "--sweep-spec", str(spec),
                           "--root", str(production), "--cluster-nodes", record["nodes"],
                           "--cpus-per-node", record["cpus_per_node"], "--gpus-per-node", record["gpus_per_node"]])
        adapter.main()
        assert sentinel.read_text() == "keep production manifest"
        assert list(production.iterdir()) == [sentinel]


def test_configuration_defaults_optional_types_and_typo_rejection(tmp_path, tune, monkeypatch):
    from test_ray_tune_sweeps import spec_file
    for flag in ("measure_stages", "measure_services"):
        base = load_and_expand_scenario(adapter.SCENARIOS["ppo"])
        assert base["experiment"][flag] is False
        enabled = load_and_expand_scenario(adapter.SCENARIOS["ppo"], overrides=[f"experiment.{flag}=true"])
        validate_experiment_scenario(enabled)
        assert enabled["experiment"][flag] is True
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=tune))
    # measure_cuda_sync is supported natively but absent from the base YAML.
    for override, error in [("experiment.measure_cuda_sync=false", None),
                            ("experiment.profiling_detail_samples=2", None),
                            ('experiment.measure_stages="true"', "requires bool"),
                            ("++experiment.measure_bogus=true", "Unknown configuration path"),
                            ("++experiment.measure_cuda_sync=1", "requires bool"),
                            ("++experiment.profiling_detail_samples=33", r"integer in \[0, 32\]")]:
        spec = spec_file(tmp_path, fixed_overrides=[override])
        monkeypatch.setattr(sys, "argv", ["adapter", "--preflight-only", "--sweep-spec", str(spec)])
        if error:
            with pytest.raises((ValueError, ScenarioError), match=error):
                adapter.main()
        else:
            adapter.main()
    spec = spec_file(tmp_path)
    monkeypatch.setattr(sys, "argv", ["adapter", "--preflight-only", "--sweep-spec", str(spec),
                       "--cluster-nodes", "1", "--cpus-per-node", "2", "--gpus-per-node", "1"])
    with pytest.raises(ValueError, match="do not fit"):
        adapter.main()


def test_invalid_profile_prevents_every_submission(matrix_env):
    env = {**matrix_env, "FAIL_PROFILE": "w16_n1g1"}
    cp = submitter(env, "--submit", "--tag", "invalid")
    assert cp.returncode != 0
    assert "PREFLIGHT FAILED: w16_n1g1" in cp.stderr and "Unknown configuration path" in cp.stderr
    assert not any(line.startswith("submit:") for line in Path(env["MATRIX_TEST_EVENTS"]).read_text().splitlines())
    manifest = Path(env["PROFILE_MATRIX_ROOT"]) / "invalid/jobs.tsv"
    assert len(manifest.read_text().splitlines()) == 1


def test_single_profile_smoke_selection(matrix_env):
    cp = submitter(matrix_env, "--dry-run", "--profile", "w04_n1g1", "--tag", "single")
    assert cp.returncode == 0, cp.stderr
    assert Path(matrix_env["MATRIX_TEST_EVENTS"]).read_text().splitlines() == ["validate:w04_n1g1"]
    with (Path(matrix_env["PROFILE_MATRIX_ROOT"]) / "single/jobs.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1 and rows[0]["profile"] == "w04_n1g1"


def test_afterok_chain_and_preserved_ids(matrix_env):
    cp = submitter(matrix_env, "--submit", "--tag", "chain")
    assert cp.returncode == 0, cp.stderr
    events = Path(matrix_env["MATRIX_TEST_EVENTS"]).read_text().splitlines()
    assert all(line.startswith("validate:") for line in events[:7])
    submitted = events[7:]
    assert len(submitted) == 7 and "--dependency" not in submitted[0]
    for index, line in enumerate(submitted[1:], 1):
        assert f"--dependency=afterok:{8000+index}" in line
        assert "--kill-on-invalid-dep=no" in line
    assert all("--partition=gpu" in line and "reserved-q1wang" not in line for line in submitted)
    with (Path(matrix_env["PROFILE_MATRIX_ROOT"]) / "chain/jobs.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [r["job_id"] for r in rows] == [str(i) for i in range(8001, 8008)]
    assert "DependencyNeverSatisfied" in cp.stdout and "scancel --state=PENDING" in cp.stdout
    # Tag reuse cannot overwrite either IDs or the failed/finished profile outputs.
    again = submitter(matrix_env, "--dry-run", "--tag", "chain")
    assert again.returncode != 0 and "already exists" in again.stderr


def test_partial_submission_retains_ids(matrix_env):
    cp = submitter({**matrix_env, "FAIL_SBATCH_AT": "3"}, "--submit", "--tag", "partial")
    assert cp.returncode != 0 and "IDs are preserved" in cp.stderr
    with (Path(matrix_env["PROFILE_MATRIX_ROOT"]) / "partial/jobs.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [r["job_id"] for r in rows] == ["8001", "8002"]


def test_summary_reports_failed_elapsed_and_preserves_successful_scaling(tmp_path, monkeypatch, capsys):
    path = adapter.REPO / "scripts/hpc/summarize_ppo_matrix.py"
    spec = importlib.util.spec_from_file_location("matrix_summary", path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    monkeypatch.setattr(summary, "__file__", str(tmp_path / "scripts/hpc/summarize_ppo_matrix.py"))
    states = {"101": "101|91|FAILED|1:0|NonZeroExitCode\n", "102": "102|0|PENDING|0:0|DependencyNeverSatisfied\n",
              "103": "103|100|COMPLETED|0:0|None\n"}
    monkeypatch.setattr(summary.subprocess, "run", lambda args, **kwargs: SimpleNamespace(stdout=states[args[5]]))
    manifest = tmp_path / "jobs.tsv"
    fields = "profile experiment_id job_id nodes gpus_per_node cpus_per_node workers envs_per_worker trials steps_per_trial".split()
    with manifest.open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(fields)
        for row in [("failed", "failed", "101"), ("blocked", "blocked", "102"), ("w16_n1g1", "success", "103")]:
            writer.writerow([*row, 1, 1, 24, 16, 8, 1, 262144])
    root = tmp_path / "outputs/ray_tune_success"
    root.mkdir(parents=True)
    trial = root / "trial"
    trial.mkdir()
    (trial / "run_summary.json").write_text(json.dumps({"perf/environment_steps_per_second": 100.,
                                                       "perf/steady_environment_steps_per_second": 120.}))
    (root / "results.json").write_text(json.dumps([{"status": "completed", "error": None,
                                                   "environment_steps": 262144, "training_output_dir": str(trial)}]))
    summary.run(manifest)
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    failed, blocked, successful = rows
    assert failed["allocation_seconds"] == "91" and failed["slurm_exit_code"] == "1:0"
    assert failed["slurm_state"] == "FAILED" and failed["missing_results"] == "True"
    assert failed["aggregate_steps_per_second"] == "" and failed["mean_native_steps_per_second"] == ""
    assert "DependencyNeverSatisfied" in blocked["failure_reason"]
    assert successful["aggregate_steps_per_second"] == "2621.44" and successful["mean_native_steps_per_second"] == "100.0"
    assert successful["speedup_vs_w16_n1g1"] == "1.0"
    assert "wall=91s" in capsys.readouterr().out
