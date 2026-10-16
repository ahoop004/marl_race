#!/usr/bin/env python3
"""Ray 2.59 orchestration of native PPO/MAPPO subprocesses on a shared filesystem.

Native loggers own metrics and W&B. Tune iterations are polling/report events;
``environment_steps`` is the comparison axis. Recovery uses run.py --resume.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "ppo": "scenarios/ppo_lap_completion_pretrain.yaml",
    "mappo": "scenarios/mappo_2v2_completion_scratch.yaml",
}


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive(path, *, wait_seconds=0):
    """Reject concurrent allocations/retries writing the same persistent output."""
    with path.open("a") as stream:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Already in use: {path.parent}") from exc
                time.sleep(.1)
        yield stream


class Tail:
    """Incremental complete-record reader; re-read only on replacement/truncation.

    CSVLogger atomically replaces CSVs when expanding their schema. Retain a
    small prefix/tail fingerprint too, to detect same-inode rewrites. Incomplete
    lines (including quoted multiline CSV records) stay pending until flushed.
    """

    def __init__(self, path, *, csv_file=False, at_end=False):
        self.path, self.csv_file = path, csv_file
        self.offset = 0
        self.identity = None
        self.fields = None
        self.prefix = self.anchor = b""
        if at_end:
            self.read(discard=True)

    def read(self, *, discard=False):
        rows = []
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                prefix = stream.read(len(self.prefix))
                stream.seek(max(0, self.offset - len(self.anchor)))
                anchor = stream.read(len(self.anchor))
                if (identity != self.identity or stat.st_size < self.offset
                        or prefix != self.prefix or anchor != self.anchor):
                    self.offset, self.fields = 0, None
                self.identity = identity
                stream.seek(self.offset)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    try:
                        if self.csv_file:
                            # strict CSV parsing waits for the end of quoted records.
                            while True:
                                try:
                                    values = next(csv.reader(io.StringIO(line.decode()), strict=True))
                                    break
                                except csv.Error:
                                    more = stream.readline()
                                    if not more or not more.endswith(b"\n"):
                                        stream.seek(start)
                                        values = None
                                        break
                                    line += more
                            if values is None:
                                break
                            if self.fields is None:
                                self.fields = values
                            elif not discard and len(values) == len(self.fields):
                                rows.append(dict(zip(self.fields, values)))
                        else:
                            value = json.loads(line)
                            if not discard:
                                rows.append(value)
                    except (ValueError, UnicodeDecodeError):
                        # A complete malformed line must not poison future polling.
                        pass
                    self.offset = stream.tell()
                stream.seek(0)
                self.prefix = stream.read(min(self.offset, 128))
                stream.seek(max(0, self.offset - 128))
                self.anchor = stream.read(min(self.offset, 128))
        except FileNotFoundError:
            pass
        return rows


def scalars(values, prefix=""):
    result = {}
    for key, value in values.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        if isinstance(value, str):
            if value in {"True", "False"}:
                value = value == "True"
            else:
                try:
                    value = float(value)
                except ValueError:
                    continue
        if isinstance(value, (bool, int, float)) and math.isfinite(value):
            result[prefix + key] = value
    return result


COUNTERS = {"environment_steps": "train/environment_steps", "learner_samples": "train/agent_steps",
            "completed_episodes": "train/completed_episodes", "updates": "train/updates",
            "optimizer_steps": "train/optimizer_steps_total"}


class Metrics:
    """Native metrics plus monotonic Tune progress aliases across recovery.

    Fresh episodes can change learner participation and episode counts during
    replay. ``train/*`` retains native values; unprefixed counters retain their
    high-water marks until replay catches up, without redefining Tune iterations.
    """
    def __init__(self, out, saved=None, *, resumed=False):
        self.updates = Tail(out / "update_metrics.csv", csv_file=True, at_end=resumed)
        self.evaluations = Tail(out / "evaluation_history.jsonl", at_end=resumed)
        self.values = dict(saved or {})
        self.eval_clock = (self.values.get("eval/environment_steps", -1),
                           self.values.get("eval/policy_version", -1))
        self.update_clock = (self.values.get("adapter/csv_environment_steps", -1),
                             self.values.get("adapter/csv_updates", -1))

    def poll(self):
        changed = False
        for row in self.updates.read():
            values = scalars(row)
            steps, updates = values.get("train/environment_steps", 0), values.get("train/updates", 0)
            if ((steps, updates) > self.update_clock
                    and steps >= self.values.get("environment_steps", -1)
                    and updates >= self.values.get("updates", -1)):
                self.update_clock = (steps, updates)
                self.values.update(values)
                self.values.update({"adapter/csv_environment_steps": steps, "adapter/csv_updates": updates})
                self.values.update({name: max(self.values.get(name, 0), int(values[key]))
                                    for name, key in COUNTERS.items() if key in values})
                changed = True
        for row in self.evaluations.read():
            clock = (row.get("environment_steps", 0), row.get("policy_version", 0))
            if clock <= self.eval_clock:
                continue
            self.eval_clock = clock
            self.values.update(scalars(row, "eval/"))
            for index, value in enumerate(row.get("selection_score", [])):
                if value is not None and math.isfinite(value):
                    self.values[f"eval/selection_score/{index}"] = value
            changed = True
        return changed

    def progress(self, progress):
        if progress["environment_steps"] < self.values.get("environment_steps", -1):
            return
        for name, key in COUNTERS.items():
            source = {"learner_samples": "agent_steps"}.get(name, name)
            if source in progress:
                self.values[key] = progress[source]
                self.values[name] = max(self.values.get(name, 0), progress[source])

    def summary(self, out):
        summary = json.loads((out / "run_summary.json").read_text())
        self.progress(summary)
        self.values.update({k: v for k, v in scalars(summary).items() if k.startswith("perf/")})


def checkpoint_payload(path, identity):
    # Reuse native schema validation; full model/task/optimizer validation occurs
    # in run.py before any training. Never fall back after a loading failure.
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from agents.common.checkpoints import read_checkpoint, RESUME_BOUNDARY
    payload = read_checkpoint(path)
    metadata, state = payload["metadata"], payload["training"]
    if not state or state.get("boundary") != RESUME_BOUNDARY:
        raise ValueError(f"Not a resumable native checkpoint: {path}")
    expected = (identity["algorithm"], identity["run_id"], identity["wandb_id"])
    actual = (metadata["algorithm"], metadata["provenance"].get("run_id"),
              metadata["provenance"].get("wandb_run_id"))
    enabled = identity["settings"]["wandb"]["mode"] != "disabled"
    if actual[:2] != expected[:2] or (enabled and actual[2] != expected[2]):
        raise ValueError(f"Checkpoint belongs to a different trial: {path}")
    if state["budget"]["total_steps"] != identity["settings"]["total_steps"]:
        raise ValueError("Checkpoint training budget differs from the original trial")
    return payload


def copy_checkpoint(source, target):
    # Native saves use atomic rename. Open once to pin that complete generation,
    # then copy bytes, never symlink the mutable latest.pt into a Ray checkpoint.
    with source.open("rb") as old, target.open("wb") as new:
        shutil.copyfileobj(old, new)
        new.flush()
        os.fsync(new.fileno())


def restore_native(checkpoint, out, attempt, identity):
    candidates = []
    if checkpoint is not None:
        with checkpoint.as_directory() as directory:
            directory = Path(directory)
            stored = json.loads((directory / "adapter.json").read_text())
            if stored["identity"] != identity:
                raise ValueError("Ray checkpoint trial identity/configuration mismatch")
            copy_checkpoint(directory / "latest.pt", attempt / "resume.pt")
            payload = checkpoint_payload(attempt / "resume.pt", identity)
            candidates.append((attempt / "resume.pt", payload))
    # Shared latest may be newer than Tune's last durable experiment-state save.
    # Validate it too; a corrupt/incompatible latest must fail visibly.
    if (out / "latest.pt").exists():
        copy_checkpoint(out / "latest.pt", attempt / "native_resume.pt")
        candidates.append((attempt / "native_resume.pt", checkpoint_payload(attempt / "native_resume.pt", identity)))
    if not candidates:
        if (out / "started.json").exists():
            raise RuntimeError(f"Trial already started but has no recovery checkpoint: {out}")
        return None
    return max(candidates, key=lambda item: item[1]["training"]["progress"]["environment_steps"])[0]


def trainer_command(settings, identity, out):
    cmd = [sys.executable, str(REPO / "run.py"), "--scenario", settings["scenarios"][identity["algorithm"]],
           "--output-dir", str(out), "--run-id", identity["run_id"],
           "--total-steps", str(settings["total_steps"]), "--seed", str(settings["seed"]),
           "--torch-threads", "1", "--no-render"]
    overrides = ["experiment.parallel_envs=2", "experiment.envs_per_worker=1", "experiment.distributed=false",
                 "algorithm.params.n_steps=1024", "algorithm.params.frames_per_batch=256",
                 "algorithm.params.batch_size=128", "algorithm.params.n_epochs=2",
                 f"checkpoint.every_steps={settings['checkpoint_every_steps']}",
                 f"checkpoint.every_updates={settings['checkpoint_every_updates'] or 'null'}",
                 f"evaluation.enabled={str(settings['evaluation']).lower()}",
                 "execution.params.device=cuda", "execution.params.collection_device=cpu", *settings["overrides"]]
    wandb = settings["wandb"]
    metadata = {"experiment_id": settings["experiment_id"], "trial_id": identity["trial_id"],
                "algorithm": identity["algorithm"], "scenario": settings["scenarios"][identity["algorithm"]],
                "git_sha": settings["git_sha"]}
    wb = {"enabled": wandb["mode"] != "disabled", "mode": wandb["mode"],
          "name": identity["run_id"], "id": identity["wandb_id"],
          "group": wandb["group"] or settings["experiment_id"], "job_type": identity["algorithm"], "tune": metadata}
    wb.update({key: wandb[key] for key in ("project", "entity") if wandb[key] is not None})
    overrides.extend(f"++wandb.{key}={json.dumps(value)}" for key, value in wb.items() if key != "tune")
    overrides.extend(f"++wandb.tune.{key}={json.dumps(value)}" for key, value in metadata.items())
    for override in overrides:
        cmd.extend(["--set", override])
    return cmd


def stop_group(process):
    pid = process.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader; a zombie otherwise keeps its group alive.
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise(parent_pid, command):
    """Watch the Ray worker even if it dies while its function thread reports.

    This process has its own session; trainers/collectors share a second process
    group. Signal handlers run here on the main thread, outside Ray's threads.
    """
    process = None

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None:
            if os.getppid() != parent_pid:
                raise SystemExit("Ray worker exited; terminating trainer process group")
            time.sleep(.5)
        return process.wait()
    finally:
        if process is not None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            stop_group(process)
            process.wait()


def trial(config):
    import ray
    from ray import tune

    settings = config["settings"]
    label, trial_id = config["algorithm"], tune.get_context().get_trial_id()
    out = Path(settings["root"]) / "trial_runs" / f"{label}_{trial_id}"
    identity = {"algorithm": label, "trial_id": trial_id, "settings": settings,
                "run_id": f"{settings['experiment_id']}_{label}_{trial_id}",
                "wandb_id": hashlib.sha256(f"{settings['root']}:{trial_id}:{label}".encode()).hexdigest()[:24]}
    out.mkdir(parents=True, exist_ok=True)
    with exclusive(out / ".lock", wait_seconds=20) as lease:
        identity_path = out / "ray_trial.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError(f"Refusing to reuse unrelated results: {out}")
        elif any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError(f"Unrecognized trial directory: {out}")
        else:
            atomic_json(identity_path, identity)
        allocated = ray.get_gpu_ids()
        if len(allocated) != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError(f"Expected one Ray GPU: {allocated}, CVD={os.getenv('CUDA_VISIBLE_DEVICES')!r}")
        attempt = Path(tempfile.mkdtemp(prefix=f"{os.getenv('SLURM_JOB_ID', 'local')}_", dir=_attempts(out)))
        checkpoint = tune.get_checkpoint()
        resume = restore_native(checkpoint, out, attempt, identity)
        input_path = out / "trial_input.yaml"
        command = trainer_command(settings, identity, out)
        if not input_path.exists():
            if resume:
                raise RuntimeError("Missing original trial_input.yaml; refusing to recompose recovery configuration")
            resolved = subprocess.run([*command, "--resolve-config"], cwd=REPO, text=True, capture_output=True)
            if resolved.returncode:
                raise RuntimeError(f"Native configuration resolution failed:\n{resolved.stderr}")
            input_path.write_text(resolved.stdout)
            atomic_json(out / "input_identity.json", {"sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()})
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != json.loads((out / "input_identity.json").read_text())["sha256"]:
            raise ValueError("Original resolved trial configuration was modified")
        # Reuse the frozen native resolved configuration, including all budgets
        # and transfer sources. Only attempt-specific tracking metadata changes.
        command = [sys.executable, str(REPO / "run.py"), "--scenario", str(input_path),
                   "--output-dir", str(out), "--run-id", identity["run_id"]]
        for key, value in {"slurm_job_id": os.getenv("SLURM_JOB_ID"), "hostname": socket.gethostname()}.items():
            command.extend(["--set", f"++wandb.tune.{key}={json.dumps(value)}"])
        if resume:
            command.extend(["--resume", str(resume)])
        metadata = {"hostname": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                    "ray_gpu_ids": [str(x) for x in allocated], "command": command,
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
        atomic_json(attempt / "attempt.json", metadata)
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "WEIGHT_SYNC_TIMEOUT": "900", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1", "PYGLET_HEADLESS": "true", "F110_SUMMARY_EVERY": "10"})
        journal = out / "ray_report_state.json"
        saved = json.loads(journal.read_text()) if journal.exists() else {}
        metrics = Metrics(out, saved, resumed=bool(resume))
        process, checkpoint_stamp = None, None
        if checkpoint is not None and (out / "latest.pt").exists():
            progress = checkpoint_payload(out / "latest.pt", identity)["training"]["progress"]
            if progress["environment_steps"] <= saved.get("checkpoint_environment_steps", -1):
                stat = (out / "latest.pt").stat()
                checkpoint_stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        base = {"algorithm": label, "hostname": metadata["hostname"], "training_output_dir": str(out)}

        def report(completed=False):
            nonlocal checkpoint_stamp
            changed = metrics.poll()
            latest = out / "latest.pt"
            stamp = None
            try:
                stat = latest.stat()
                stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                pass
            if completed:
                metrics.summary(out)
            if stamp is not None and stamp != checkpoint_stamp:
                with tempfile.TemporaryDirectory(prefix=".ray-checkpoint-", dir=out) as directory:
                    directory = Path(directory)
                    copy_checkpoint(latest, directory / "latest.pt")
                    payload = checkpoint_payload(directory / "latest.pt", identity)
                    metrics.progress(payload["training"]["progress"])
                    metrics.values["checkpoint_environment_steps"] = payload["training"]["progress"]["environment_steps"]
                    atomic_json(directory / "adapter.json", {"identity": identity})
                    # Include frozen config for auditing; persistent native outputs
                    # remain required when restoring this shared-filesystem experiment.
                    shutil.copyfile(input_path, directory / "trial_input.yaml")
                    send(directory, completed)
                    checkpoint_stamp = stamp
            elif changed or completed:
                send(None, completed)

        def send(directory, completed):
            values = {**metrics.values, **base, "status": "completed" if completed else "running"}
            # Write ahead: replay may skip an in-flight report, but never emits a
            # lower decision/update clock after a worker or driver failure.
            atomic_json(journal, metrics.values)
            tune.report(values, checkpoint=tune.Checkpoint.from_directory(str(directory)) if directory else None)

        started = time.monotonic()
        timeout = float(os.getenv("RAY_TUNE_TRIAL_TIMEOUT", settings["trial_timeout"]))
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Trial timeout must be finite and >=0 (0 disables it)")
        try:
            atomic_json(out / "started.json", metadata)
            with (attempt / "train.log").open("w") as logfile:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise",
                                            str(os.getpid()), *command], cwd=REPO, env=env,
                                           stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True,
                                           # Keep the output lease until orphan cleanup finishes.
                                           pass_fds=(lease.fileno(),))
                while process.poll() is None:
                    report()
                    if timeout and time.monotonic() - started > timeout:
                        raise TimeoutError(f"{label} exceeded {timeout}s; inspect {attempt / 'train.log'}")
                    time.sleep(settings["poll_interval"])
                if process.returncode != 0:
                    report()
                    raise RuntimeError(f"{label} trainer exited {process.returncode}; inspect {attempt / 'train.log'}")
                summary = json.loads((out / "run_summary.json").read_text())
                if summary["environment_steps"] < settings["total_steps"] or not (out / "final.pt").is_file():
                    report()
                    raise RuntimeError(f"{label}: original training budget not reached; inspect {attempt / 'train.log'}")
                if not (out / "latest.pt").is_file():
                    raise RuntimeError(f"{label}: no native recovery checkpoint")
                report(completed=True)
        finally:
            if process is not None:
                stop_group(process)
                process.wait()


def _attempts(out):
    path = out / "attempts"
    path.mkdir(exist_ok=True)
    return path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=os.getenv("EXPERIMENT_ID", f"slurm_{os.getenv('SLURM_JOB_ID', 'local')}"))
    parser.add_argument("--root", default=os.getenv("RAY_TUNE_JOB_ROOT"), help="Persistent shared experiment root")
    parser.add_argument("--restore", action="store_true", default=os.getenv("RESTORE", "0") == "1")
    parser.add_argument("--mode", choices=("ppo", "mappo", "both"), default=os.getenv("TRIAL_MODE", "ppo"))
    parser.add_argument("--total-steps", type=int, default=int(os.getenv("RAY_TUNE_TOTAL_STEPS", "4096")))
    parser.add_argument("--checkpoint-every-steps", type=int, default=int(os.getenv("RAY_TUNE_CHECKPOINT_STEPS", "2048")))
    parser.add_argument("--checkpoint-every-updates", type=int, default=None)
    parser.add_argument("--trial-timeout", type=float, default=float(os.getenv("RAY_TUNE_TRIAL_TIMEOUT", "1200")), help="Seconds per attempt; 0 disables")
    parser.add_argument("--poll-interval", type=float, default=3)
    parser.add_argument("--max-failures", type=int, default=0, help="Tune automatic retries per trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default=os.getenv("RAY_TUNE_WANDB_MODE", "disabled"))
    for key in ("project", "entity", "group"):
        parser.add_argument(f"--wandb-{key}", default=os.getenv(f"RAY_TUNE_WANDB_{key.upper()}"))
    parser.add_argument("--evaluation", action="store_true", default=os.getenv("RAY_TUNE_EVALUATION", "0") == "1")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Native Hydra override (repeatable)")
    for label, path in SCENARIOS.items():
        parser.add_argument(f"--{label}-scenario", default=path)
    return parser.parse_args()


def main():
    args = arguments()
    import ray
    from ray import tune

    if not args.experiment_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.experiment_id):
        raise ValueError("experiment-id must contain only letters, digits, underscores or hyphens")
    if (args.total_steps <= 0 or args.checkpoint_every_steps <= 0 or args.max_failures < 0
            or args.checkpoint_every_updates is not None and args.checkpoint_every_updates <= 0
            or not math.isfinite(args.poll_interval) or args.poll_interval <= 0
            or not math.isfinite(args.trial_timeout) or args.trial_timeout < 0):
        raise ValueError("Budgets/intervals must be positive; timeout and max-failures must be >=0")
    root = Path(args.root or REPO / "outputs" / f"ray_tune_{args.experiment_id}").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / ".experiment.lock"):
        manifest = root / "experiment.json"
        experiment_path = root / "tune_results" / "independent_trials"
        if args.restore:
            settings = json.loads(manifest.read_text())
            if settings["experiment_id"] != args.experiment_id or settings["root"] != str(root):
                raise ValueError("Restore identity/path differs from the saved experiment")
            if not tune.Tuner.can_restore(str(experiment_path)):
                raise RuntimeError(f"No restorable Tune experiment at {experiment_path}")
        else:
            if manifest.exists() or experiment_path.exists() or (root / "trial_runs").exists():
                raise FileExistsError(f"Experiment already exists; use --restore: {root}")
            settings = {"experiment_id": args.experiment_id, "root": str(root), "mode": args.mode,
                        "total_steps": args.total_steps, "seed": args.seed,
                        "checkpoint_every_steps": args.checkpoint_every_steps,
                        "checkpoint_every_updates": args.checkpoint_every_updates,
                        "trial_timeout": args.trial_timeout, "poll_interval": args.poll_interval,
                        "evaluation": args.evaluation, "overrides": args.overrides,
                        "scenarios": {label: str((REPO / getattr(args, f"{label}_scenario")).resolve()) for label in SCENARIOS},
                        "wandb": {key: getattr(args, f"wandb_{key}") for key in ("mode", "project", "entity", "group")},
                        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
            atomic_json(manifest, settings)
        algorithms = ["ppo", "mappo"] if settings["mode"] == "both" else [settings["mode"]]
        ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if ray.cluster_resources().get("GPU", 0) >= 2 and len([n for n in ray.nodes() if n["Alive"]]) == 2:
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f"Two-node Ray cluster not ready: {ray.cluster_resources()}")
            trainable = tune.with_resources(trial, resources={"cpu": 6, "gpu": 1})
            if args.restore:
                tuner = tune.Tuner.restore(str(experiment_path), trainable=trainable,
                                           resume_unfinished=True, resume_errored=True, restart_errored=False)
            else:
                tuner = tune.Tuner(trainable, param_space={"algorithm": tune.grid_search(algorithms), "settings": settings},
                                   tune_config=tune.TuneConfig(max_concurrent_trials=len(algorithms)),
                                   run_config=tune.RunConfig(name="independent_trials", storage_path=str(root / "tune_results"),
                                                            progress_reporter=tune.CLIReporter(
                                                                metric_columns=list(COUNTERS), parameter_columns=["algorithm"]),
                                                            failure_config=tune.FailureConfig(max_failures=args.max_failures),
                                                            checkpoint_config=tune.CheckpointConfig(num_to_keep=2), verbose=1))
            results = []
            for result in tuner.fit():
                metrics = result.metrics or {}
                results.append({key: metrics.get(key) for key in ("algorithm", "hostname", "status", "environment_steps", "training_output_dir")})
                results[-1].update(tune_result_path=str(result.path), error=str(result.error) if result.error else None)
            atomic_json(root / "results.json", results)
            print(json.dumps(results, indent=2), flush=True)
            if len(results) != len(algorithms) or any(r["error"] or r["status"] != "completed" for r in results):
                raise RuntimeError("One or more Ray Tune trials failed")
        finally:
            ray.shutdown()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--supervise":
        sys.exit(supervise(int(sys.argv[2]), sys.argv[3:]))
    main()
