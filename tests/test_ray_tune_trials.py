"""Focused adapter checks without a Ray installation, W&B network, or GPUs."""
from contextlib import nullcontext
from copy import deepcopy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from agents.common.checkpoints import read_checkpoint
from loggers.csv_logger import CSVLogger
from training.hooks import CSVHook, CheckpointHook
from test_torchrl_ppo import Capture, trainer

spec = importlib.util.spec_from_file_location("ray_tune_trials", Path(__file__).parents[1] / "scripts/ray_tune_trials.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def settings(tmp_path, mode="online"):
    return {"root": str(tmp_path), "experiment_id": "recovery", "mode": "both", "total_steps": 9,
            "seed": 42, "checkpoint_every_steps": 4, "checkpoint_every_updates": None,
            "evaluation": False, "overrides": [], "scenarios": adapter.SCENARIOS,
            "wandb": {"mode": mode, "group": None, "project": "test", "entity": "test"},
            "git_sha": "original-sha", "trial_timeout": 30, "poll_interval": .001}


def test_incremental_csv_partial_records_replacement_and_rewrite(tmp_path):
    path = tmp_path / "update_metrics.csv"
    tail = adapter.Tail(path, csv_file=True)
    assert tail.read() == []
    path.write_text('step,loss\n1,0.5\n2,')
    assert tail.read() == [{"step": "1", "loss": "0.5"}]
    assert tail.read() == []
    with path.open("a") as stream:
        stream.write('0.4\n3,"multi\n')
    assert tail.read() == [{"step": "2", "loss": "0.4"}]
    with path.open("a") as stream:
        stream.write('line"\n')
    assert tail.read() == [{"step": "3", "loss": "multi\nline"}]
    assert tail.read() == []
    replacement = tmp_path / "replacement"
    replacement.write_text('step,loss,entropy\n4,0.3,0.8\n')
    replacement.replace(path)
    assert tail.read() == [{"step": "4", "loss": "0.3", "entropy": "0.8"}]
    path.write_text('step,loss\n5,0.2\n')  # truncate in place
    assert tail.read() == [{"step": "5", "loss": "0.2"}]
    path.write_text('step,loss\n6,0.1\n')  # same size, same inode
    assert tail.read() == [{"step": "6", "loss": "0.1"}]


def update(out, steps, updates):
    path = out / "update_metrics.csv"
    exists = path.exists()
    with path.open("a", newline="") as stream:
        fields = [*adapter.COUNTERS.values(), "train/loss", "perf/elapsed_seconds", "perf/environment_steps_per_second"]
        writer = csv.DictWriter(stream, fields)
        if not exists:
            writer.writeheader()
        writer.writerow(dict(zip(fields, [steps, steps * 2, updates, updates, updates * 3, .1, 2., 4.])))


def test_metrics_evaluation_and_replay_keep_distinct_monotonic_counters(tmp_path):
    metrics = adapter.Metrics(tmp_path)
    update(tmp_path, 8, 2)
    assert metrics.poll()
    assert metrics.values["learner_samples"] == 16
    assert metrics.values["optimizer_steps"] == 6
    assert metrics.values["updates"] == 2
    assert metrics.values["train/loss"] == .1
    assert metrics.values["perf/environment_steps_per_second"] == 4
    evaluation = tmp_path / "evaluation_history.jsonl"
    record = {"environment_steps": 8, "policy_version": 2, "team_both_finished_rate": .5,
              "collision_rate": .1, "mean_valid_lap_time_s": 15., "mean_net_progress": .8,
              "selection_score": [.5, .8, None], "episode_results": [{"reward": 1}]}
    evaluation.write_text(json.dumps(record)[:-1])
    assert not metrics.poll()
    with evaluation.open("a") as stream:
        stream.write("}\n")
    assert metrics.poll()
    assert metrics.values["eval/team_both_finished_rate"] == .5
    assert metrics.values["eval/selection_score/1"] == .8
    saved = deepcopy(metrics.values)
    resumed = adapter.Metrics(tmp_path, saved, resumed=True)
    update(tmp_path, 4, 1)  # rollback and replay: suppressed
    update(tmp_path, 8, 2)
    assert not resumed.poll()
    update(tmp_path, 12, 3)
    assert resumed.poll()
    assert resumed.values["environment_steps"] == 12
    assert resumed.values["learner_samples"] == 24
    assert not resumed.poll()
    resumed.progress({"environment_steps": 13, "agent_steps": 23, "completed_episodes": 1,
                      "updates": 3, "optimizer_steps": 8})
    assert resumed.values["environment_steps"] == 13
    assert resumed.values["learner_samples"] == 24  # Tune high-water progress
    assert resumed.values["train/agent_steps"] == 23  # Native recovered counter


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_native_configuration_roundtrip_and_wandb_metadata(tmp_path, algorithm, mode):
    from core.scenario import load_and_expand_scenario
    from application.cli import apply_cli_overrides
    from application.configuration import validate_experiment_scenario
    from omegaconf import OmegaConf

    cfg = settings(tmp_path, mode)
    identity = {"algorithm": algorithm, "trial_id": algorithm, "run_id": f"recovery_{algorithm}", "wandb_id": f"unique_{algorithm}"}
    command = adapter.trainer_command(cfg, identity, tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", command[1:])
        from application.cli import parse_args
        args = parse_args()
    resolved = apply_cli_overrides(load_and_expand_scenario(args.scenario, validate=False, overrides=args.parameter_overrides), args)
    validate_experiment_scenario(resolved)
    wb = resolved["wandb"]
    assert wb["id"] == identity["wandb_id"] and wb["name"] == identity["run_id"]
    assert wb["group"] == "recovery" and wb["enabled"] == (mode != "disabled")
    assert wb["tune"]["trial_id"] == algorithm and wb["tune"]["git_sha"] == "original-sha"
    frozen = tmp_path / "input.yaml"
    OmegaConf.save(OmegaConf.create(resolved), frozen)
    reloaded = load_and_expand_scenario(str(frozen), validate=False)
    assert {k: v for k, v in resolved.items() if k != "_configuration"} == {
        k: v for k, v in reloaded.items() if k != "_configuration"}
    assert reloaded["experiment"]["total_steps"] == 9


@pytest.mark.parametrize("sweep", [False, True])
def test_trial_registers_immutable_native_checkpoint_and_recovers_new_allocation(tmp_path, monkeypatch, sweep):
    cfg = settings(tmp_path)
    entry = {"configuration_id": "point", "sample_index": 0, "replicate": 0, "seed": 42,
             "parameters": {"algorithm.params.learning_rate": .0003}, "schedule": {"lr_schedule": "constant"}}
    if sweep:
        cfg["sweep"] = {"name": "sweep", "search": {"method": "grid"}, "sweep_seed": 5, "evaluation": {}}
    trial_config = {"algorithm": "ppo", "settings": cfg, **({"sweep_trial": entry} if sweep else {})}
    reports, resume_checkpoint, calls = [], None, []

    class Checkpoint:
        def __init__(self, directory):
            self.directory = Path(directory)

        @classmethod
        def from_directory(cls, directory):
            return cls(directory)

        def as_directory(self):
            return nullcontext(self.directory)

    def report(values, checkpoint=None):
        nonlocal resume_checkpoint
        if checkpoint:
            destination = tmp_path / f"registered_{len(reports)}"
            shutil.copytree(checkpoint.directory, destination)
            resume_checkpoint = Checkpoint(destination)
        reports.append(values)

    tune = SimpleNamespace(get_context=lambda: SimpleNamespace(get_trial_id=lambda: "unique"),
                           get_checkpoint=lambda: resume_checkpoint, Checkpoint=Checkpoint, report=report)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=tune, get_gpu_ids=lambda: [0]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "first")
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="native resolved config\n", returncode=0))

    class Process:
        pid = 99999999

        def __init__(self, command, **kwargs):
            self.returncode = 0
            out = tmp_path / "trial_runs/ppo_unique"
            identity = json.loads((out / "ray_trial.json").read_text())
            provenance = {"run_id": identity["run_id"], "wandb_run_id": identity["wandb_id"]}
            capture = Capture(False)
            native, _ = trainer(capture, **({"learning_rate": .0003} if sweep else {}))
            logger = CSVLogger(str(out), append="--resume" in command)
            configuration = {"experiment": {"seed": 42, "total_steps": 9},
                             "algorithm": {"params": {"learning_rate": .0003}}} if sweep else None
            native.hooks.extend([CSVHook(logger), CheckpointHook(native.agent, out, provenance=provenance,
                                                               configuration=configuration, save_every_steps=4)])
            if "--resume" in command:
                state, _ = native.agent.resume_training(command[command.index("--resume") + 1])
                native.restore_training_state(state)
                assert native.agent.optimizer.state
                assert native.collected_steps == 4 and native._updates == 1
                assert state["budget"]["total_steps"] == 9
            else:
                native.stop_after_steps = 4
            native.train(total_steps=9)
            calls.append(native.collected_steps)
            self.returncode = 0 if "--resume" in command else -signal.SIGTERM

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(adapter.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="trainer exited"):
        adapter.trial(trial_config)
    registered = resume_checkpoint.directory
    before = (registered / "latest.pt").read_bytes()
    assert read_checkpoint(registered / "latest.pt")["training"]["progress"]["environment_steps"] == 4
    monkeypatch.setenv("SLURM_JOB_ID", "second")
    adapter.trial(trial_config)
    out = tmp_path / "trial_runs/ppo_unique"
    assert calls == [4, 9]
    assert (registered / "latest.pt").read_bytes() == before
    assert reports[-1]["status"] == "completed" and reports[-1]["environment_steps"] == 9
    assert [r["environment_steps"] for r in reports] == sorted(r["environment_steps"] for r in reports)
    assert len(list((out / "attempts").iterdir())) == 2
    assert read_checkpoint(out / "latest.pt")["metadata"]["provenance"]["wandb_run_id"] == json.loads((out / "ray_trial.json").read_text())["wandb_id"]
    different = deepcopy(cfg)
    different["total_steps"] = 10
    with pytest.raises(ValueError, match="unrelated results"):
        adapter.trial({"algorithm": "ppo", "settings": different})
    identity = json.loads((out / "ray_trial.json").read_text())
    broken = deepcopy(identity)
    broken["settings"]["total_steps"] = 10
    with pytest.raises(ValueError, match="budget"):
        adapter.checkpoint_payload(out / "latest.pt", broken)
    if sweep:
        broken = deepcopy(identity)
        broken["settings"]["sweep_trial"]["parameters"]["algorithm.params.learning_rate"] = .001
        with pytest.raises(ValueError, match="hyperparameter differs"):
            adapter.checkpoint_payload(out / "latest.pt", broken)
    (out / "latest.pt").write_bytes(b"corrupt")
    with pytest.raises(Exception):
        adapter.restore_native(resume_checkpoint, out, tmp_path, identity)


def test_restore_uses_saved_settings_and_resume_flags(tmp_path, monkeypatch):
    cfg = settings(tmp_path, "disabled")
    monkeypatch.setattr(adapter, "current_revision", lambda: cfg["git_sha"])
    adapter.atomic_json(tmp_path / "experiment.json", cfg)
    restored = []

    class Tuner:
        can_restore = staticmethod(lambda path: True)

        @staticmethod
        def restore(path, **kwargs):
            restored.append(kwargs)
            # A completed trial is returned by Tune without invoking trainable.
            return SimpleNamespace(fit=lambda: [SimpleNamespace(metrics={"algorithm": "ppo", "status": "completed"}, path="ppo", error=None),
                                               SimpleNamespace(metrics={"algorithm": "mappo", "status": "completed"}, path="mappo", error=None)])

    tune = SimpleNamespace(Tuner=Tuner, with_resources=lambda fn, resources: fn)
    ray = SimpleNamespace(tune=tune, init=lambda **kw: None, shutdown=lambda: None,
                          cluster_resources=lambda: {"GPU": 2}, nodes=lambda: [{"Alive": True}] * 2)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv("RAY_ADDRESS", "test")
    monkeypatch.setattr(sys, "argv", ["adapter", "--root", str(tmp_path), "--experiment-id", "recovery", "--restore", "--total-steps", "999"])
    adapter.main()
    assert restored[0]["resume_unfinished"] and restored[0]["resume_errored"]
    assert not restored[0]["restart_errored"]
    assert "param_space" not in restored[0]
    assert json.loads((tmp_path / "experiment.json").read_text())["total_steps"] == 9


@pytest.mark.parametrize("failure", ["signal", "worker_exit"])
def test_supervisor_terminates_trainer_group(tmp_path, failure):
    # Run a real process tree without any training or Ray dependency.
    marker = tmp_path / "pid"
    command = [sys.executable, "-c", f"import os,time,pathlib; pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"]
    launch = [sys.executable, str(Path(adapter.__file__)), "--supervise", str(os.getpid()), *command]
    if failure == "worker_exit":
        worker_code = (
            f"import os,subprocess,time,pathlib; command={launch!r}; command[3]=str(os.getpid()); "
            f"subprocess.Popen(command, start_new_session=True); marker=pathlib.Path({str(marker)!r})\n"
            "while not marker.exists(): time.sleep(.05)\n"
        )
        supervisor = subprocess.Popen([sys.executable, "-c", worker_code], start_new_session=True)
    else:
        supervisor = subprocess.Popen(launch, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists()
        pid = int(marker.read_text())
        if failure == "signal":
            supervisor.terminate()
        supervisor.wait(timeout=20)
        deadline = time.monotonic() + 5
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(.05)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGKILL)
            supervisor.wait()


@pytest.mark.parametrize("saved_id,configured_id,expected", [(None, "trial-id", ("trial-id", "allow")),
                                                              ("saved-id", "trial-id", ("saved-id", "must")),
                                                              (None, None, (None, None))])
def test_application_wandb_identity_keeps_native_resume_precedence(tmp_path, monkeypatch, saved_id, configured_id, expected):
    from contextlib import ExitStack
    from application import train

    class StopSetup(Exception):
        pass

    observed = []
    scenario = {"agents": {"policy": {"algorithm": "ppo"}}, "experiment": {}, "environment": {},
                "checkpoint": {}, "wandb": {"enabled": True, "id": configured_id}}
    if saved_id:
        scenario["checkpoint"]["resume"] = "existing.pt"
        monkeypatch.setattr(train, "resolve_checkpoint_path", lambda *a, **kw: tmp_path / "existing.pt")
        monkeypatch.setattr(train, "read_checkpoint", lambda *a: {"metadata": {"provenance": {"wandb_run_id": saved_id}}, "training": {}})
        monkeypatch.setattr(train, "validate_resume_experiment", lambda *a: None)

    def logger(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(finish=lambda: None)

    def stop(*args, **kwargs):
        raise StopSetup

    monkeypatch.setattr(train, "WandbLogger", logger)
    monkeypatch.setattr(train, "create_race_task", stop)
    args = SimpleNamespace(checkpoint=None, run_id="native-run", output_dir=str(tmp_path))
    with ExitStack() as resources, pytest.raises(StopSetup):
        train._run_training(scenario, args, None, tmp_path, SimpleNamespace(policy_agents=["policy"]), resources)
    assert (observed[0].get("id"), observed[0].get("resume")) == expected
