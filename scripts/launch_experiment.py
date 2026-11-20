"""Launch independent experiments; training remains owned by run.py.

Work items are ordered scenario entries mapping output names to Hydra overrides.
A root manifest freezes every item before any training, including queued items.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import uuid

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.configuration import compose_configuration
from core.scenario import load_and_expand_scenario
from application.configuration import validate_experiment_scenario


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def task_identity(independent_cpu=False):
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    size = positive(int(os.environ.get("SLURM_NTASKS", "1")), "Slurm task count")
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    if not 0 <= rank < size:
        raise ValueError("Invalid Slurm task rank")
    if independent_cpu:
        if nodes != size or int(os.environ.get("SLURM_LOCALID", "0")) != 0:
            raise ValueError("Independent CPU experiments require exactly one Slurm task per node")
    elif nodes != 1:
        raise ValueError("Use the independent CPU entry point for multiple nodes")
    return rank, size, sorted(os.sched_getaffinity(0))


def work_items(config):
    launch = config.get("launch", {})
    if launch.get("type", "independent") not in {"independent", "continual"}:
        raise ValueError("launch.type must be independent or continual")
    items = launch.get("work_items", {"train": []})
    if not isinstance(items, dict) or not items:
        raise ValueError("launch.work_items must be a nonempty mapping of names to Hydra override lists")
    for name, overrides in items.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid work-item directory name: {name!r}")
        if not isinstance(overrides, list) or any(not isinstance(v, str) for v in overrides):
            raise ValueError(f"{name}: work-item overrides must be a list of strings")
    return items


def batch_error(config, workers, envs):
    """Avoid TorchRL collection rounding and PPO update/minibatch overshoot."""
    slots = workers * envs
    for agent in config["agents"].values():
        if agent.get("algorithm") != "ppo":
            continue
        params = {**config["training_defaults"], **agent.get("params", {})}
        frames = positive(params.get("frames_per_batch", 256), "frames_per_batch")
        target = positive(params["n_steps"], "n_steps")
        batch = positive(params["batch_size"], "batch_size")
        if frames % slots or target % frames or target % batch:
            return (f"{workers} workers x {envs} environments = {slots} slots; "
                    f"frames_per_batch={frames} must divide into whole slot steps, "
                    f"and n_steps={target} must be divisible by frames_per_batch and batch_size={batch}")
    return None


def resolve_collectors(config, cpu_count, *, frozen=False):
    policy = config.get("execution", {}).get("collector", {})
    # Historical frozen experiments predate the four-CPU policy.
    reserved = policy.get("reserved_cpus", 1)
    positive(reserved, "execution.collector.reserved_cpus")
    if config["experiment"].get("async_evaluation"):
        reserved = max(reserved, config["experiment"].get("torch_threads", 1) + 1)
    capacity = cpu_count - reserved
    exp = config["experiment"]
    envs = exp["envs_per_worker"] if frozen else policy.get("envs_per_worker", exp["envs_per_worker"])
    positive(envs, "envs_per_worker")
    workers = exp["parallel_envs"] if frozen else policy.get("workers", exp["parallel_envs"])
    if workers is None:
        workers = next((n for n in range(capacity, 0, -1) if batch_error(config, n, envs) is None), None)
        if workers is None:
            raise ValueError(f"No exact PPO collector layout fits {cpu_count} CPUs with {reserved} reserved. "
                             "Set an explicit envs-per-worker count or adjust the experiment batch configuration.")
        print(f"Collector: {cpu_count} affinity CPUs, {reserved} reserved; "
              f"using {workers} workers x {envs} environments (exact batch alignment)", file=sys.stderr, flush=True)
    positive(workers, "collector workers")
    if workers > capacity:
        raise ValueError(f"{'Frozen collector requires' if frozen else 'Collector requests'} {workers} workers "
                         f"+ {reserved} reserved CPUs, but task affinity has {cpu_count}. "
                         "Allocate more CPUs/task; recovery never changes the frozen layout.")
    # Legacy snapshots may have intentionally rounded batches; keep them intact.
    if not frozen and (error := batch_error(config, workers, envs)):
        raise ValueError(error + ". Choose compatible workers/envs or explicitly edit the scenario batch settings; "
                         "the launcher does not change optimization targets.")
    backend = exp.get("collection_backend", "process")
    params = config["training_defaults"]
    if backend == "process" and envs > 1 and params.get("collection_device") not in {None, "cpu"}:
        raise ValueError("TorchRL grouped process collection requires collection_device=cpu or null")
    if exp.get("distributed"):
        raise ValueError("Use the existing distributed launchers for experiment.distributed=true")
    if not frozen:
        exp.update(parallel_envs=workers, envs_per_worker=envs)
    budget = exp.get("total_steps")
    positive(budget if budget is not None else exp.get("episodes"), "training budget")
    validate_experiment_scenario(config)
    return config


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_code_sha256():
    """Identify training code, including dirty edits; configs are frozen separately."""
    digest = hashlib.sha256()
    for path in sorted([REPO / "run.py", *(REPO / "src").rglob("*.py")]):
        digest.update(path.relative_to(REPO).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def validate_revision(data, *, allow_code_change=False):
    """Retired wrappers and renamed scenarios do not change frozen training code."""
    expected = data.get("source_code_sha256")
    if expected and expected == source_code_sha256():
        return
    # Older clean manifests can still prove equivalence against their Git revision.
    if not expected and data.get("git_commit") and data.get("git_dirty") is False and not data.get("legacy_import"):
        diff = subprocess.run(["git", "diff", "--quiet", data["git_commit"], "--", "src", "run.py"], cwd=REPO)
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", "src", "run.py"],
                                   cwd=REPO, check=True, capture_output=True, text=True).stdout
        if diff.returncode == 0 and not untracked.strip():
            return
    message = ("Training source changed or revision compatibility cannot be proven "
               f"(original commit={data.get('git_commit')}, dirty={data.get('git_dirty')}). "
               "Restore the original training code, or audit compatibility and explicitly use --allow-code-change. "
               "Frozen configurations and W&B identities will be retained.")
    if not allow_code_change:
        raise ValueError(message)
    print("WARNING: " + message + " Proceeding with explicit code-change override.", file=sys.stderr, flush=True)


def atomic_yaml(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    temporary.replace(path)


def cli_overrides(args):
    overrides = list(args.overrides)
    for value, key in ((args.seed, "experiment.seed"), (args.profile, "continual"),
                       (args.steps_per_task, "continual.steps_per_task"),
                       (args.workers, "execution.collector.workers"),
                       (args.envs_per_worker, "execution.collector.envs_per_worker"),
                       (args.reserved_cpus, "execution.collector.reserved_cpus")):
        if value is not None:
            overrides.append(f"{key}={value}")
    return overrides


def prepare_configs(args, cpu_count):
    """Resolve items separately so Hydra projects group choices into training fields."""
    overrides = cli_overrides(args)
    template = compose_configuration(args.scenario, overrides)
    items, configs = work_items(template), {}
    sources = set()
    for name, item_overrides in items.items():
        config = load_and_expand_scenario(args.scenario, overrides=[*overrides, *item_overrides])
        source = args.source or config["adaptation"].get("checkpoint")
        if source:
            source = Path(source).expanduser().resolve()
            sources.add(str(source))
            snapshot = str(args.root / "source.pt")
            # Compose again to propagate the snapshot through all interpolations.
            config = load_and_expand_scenario(args.scenario, overrides=[*overrides, *item_overrides,
                f"adaptation.checkpoint={json.dumps(snapshot)}",
                f"paths.pretrained_checkpoint={json.dumps(snapshot)}"])
            config["experiment"]["checkpoint"] = snapshot
        config["environment"]["render"] = False
        config["checkpoint"]["resume"] = None
        config = resolve_collectors(config, cpu_count)
        seed = config["experiment"]["seed"]
        run_id = f"{args.root.name}_{name}_s{seed}"
        config.setdefault("wandb", {}).update(group=args.root.name, name=run_id, id=uuid.uuid4().hex)
        config["_launch"] = {"run_id": run_id, "work_item": name, "cpu_count": cpu_count}
        configs[name] = config
    if len(sources) > 1:
        raise ValueError("Work items must share a source checkpoint; use separate roots for different sources")
    return configs, next(iter(sources), None)


def prepare_manifest(args, cpu_count):
    root, manifest = args.root, args.root / "launch_manifest.yaml"
    if args.resume and not root.is_dir():
        raise ValueError(f"Missing recovery root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    # Serialize initialization across all Slurm tasks; queued items freeze now too.
    with (root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            data = yaml.safe_load(manifest.read_text())
            if not args.resume and (not os.environ.get("SLURM_JOB_ID")
                    or data["allocation"] != os.environ["SLURM_JOB_ID"]):
                raise ValueError(f"Existing experiment {root}; use --resume")
        else:
            existing = set(p.name for p in root.iterdir()) - {".launch.lock", ".allocation.lock", "source.pt"}
            if args.resume:
                # Import old launch roots only if every item has an original snapshot.
                # Never reconstruct missing historical settings from today's scenario.
                validate_revision({}, allow_code_change=getattr(args, "allow_code_change", False))
                if args.scenario is None:
                    raise ValueError("Legacy root has no manifest; specify its original --scenario to identify work items")
                scenario = args.scenario
                if (Path(scenario).name == "ppo_generated_lora_20k.yaml"
                        and not (REPO / scenario).exists()):
                    scenario = "scenarios/ppo_lora_families.yaml"
                template = compose_configuration(scenario)
                configs = {}
                for name in work_items(template):
                    frozen = root / name / "launch_config.yaml"
                    if not frozen.is_file():
                        raise ValueError(f"Legacy recovery missing {frozen}; cannot reconstruct historical settings")
                    configs[name] = yaml.safe_load(frozen.read_text())
                source = str(root / "source.pt") if (root / "source.pt").exists() else None
                legacy = True
            else:
                if existing:
                    raise ValueError(f"{root}: existing experiment data without a manifest; use --resume or a new root")
                configs, source = prepare_configs(args, cpu_count)
                legacy = False
            source_record = None
            if source:
                original = Path(source)
                snapshot = root / "source.pt"
                if snapshot.exists() and snapshot.resolve() != original.resolve():
                    if sha256(snapshot) != sha256(original):
                        raise ValueError(f"Existing source snapshot differs from {original}")
                elif not snapshot.exists():
                    temporary = root / ".source.pt.tmp"
                    shutil.copyfile(original, temporary)
                    temporary.chmod(0o444)
                    temporary.replace(snapshot)
                snapshot.chmod(snapshot.stat().st_mode & ~0o222)
                source_record = {"path": str(snapshot), "original": str(original), "sha256": sha256(snapshot)}
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                                 capture_output=True, text=True).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, check=True,
                                        capture_output=True, text=True).stdout.strip())
            data = dict(version=1, allocation=os.environ.get("SLURM_JOB_ID"), root=str(root),
                        git_commit=None if legacy else git, git_dirty=None if legacy else dirty,
                        source_code_sha256=None if legacy else source_code_sha256(), legacy_import=legacy,
                        source=source_record, items=configs)
            atomic_yaml(manifest, data)
        validate_revision(data, allow_code_change=getattr(args, "allow_code_change", False))
        if data["root"] != str(root):
            raise ValueError("Recovery root moved; frozen checkpoint and output paths require the original location")
        if data["source"] and sha256(Path(data["source"]["path"])) != data["source"]["sha256"]:
            raise ValueError("Frozen source checkpoint content changed; restore the original snapshot")
        for name, config in data["items"].items():
            source = config["adaptation"].get("checkpoint")
            if source and (data["source"] is None or source != data["source"]["path"]):
                raise ValueError(f"{name}: frozen source path differs from the root snapshot")
        return data


def completed(out, config):
    if not (out / "final.pt").is_file():
        return False
    if config.get("continual"):
        results = out / "continual_results.json"
        return results.is_file() and json.loads(results.read_text()).get("complete") is True
    summary = out / "run_summary.json"
    if not summary.is_file():
        return False
    summary = json.loads(summary.read_text())
    budget = config["experiment"].get("total_steps")
    key = "environment_steps" if budget is not None else "completed_episodes"
    if int(summary.get(key, 0)) < (budget or config["experiment"]["episodes"]):
        return False
    if config["experiment"].get("async_evaluation"):
        from training.selection import verify_selected_checkpoint

        try:
            verify_selected_checkpoint(out, config)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, pickle.UnpicklingError) as exc:
            print(f"{out.name}: validation selection incomplete: {exc}", file=sys.stderr, flush=True)
            return False
    return True


def verify_cpu_item(config, affinity):
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", len(affinity)))
    if len(affinity) < cpus:
        raise ValueError(f"Slurm task requested {cpus} CPUs but has only {len(affinity)} in its affinity")
    resolve_collectors(deepcopy(config), cpus, frozen=True)
    experiment = config["experiment"]
    if (not experiment.get("async_evaluation") or experiment.get("async_evaluation_device", "cpu") != "cpu"
            or experiment.get("torch_threads") != 1 or experiment.get("distributed")):
        raise ValueError("Independent CPU tasks require asynchronous CPU evaluation and one PyTorch thread")
    for agent in config["agents"].values():
        if agent.get("algorithm") == "ppo":
            params = {**config["training_defaults"], **agent.get("params", {})}
            if params.get("device") != "cpu" or params.get("collection_device") != "cpu":
                raise ValueError("Independent CPU tasks require CPU learner and collector devices")
    import torch

    if torch.cuda.is_available():
        raise ValueError("The independent CPU allocation exposes a CUDA device")


def validate_gpu(config):
    """CUDA indices remain relative to Slurm's task-local visibility."""
    device = config["training_defaults"].get("device", "cpu")
    if str(device).startswith("cuda"):
        import torch
        count = torch.cuda.device_count()
        if not count or ("SLURM_PROCID" in os.environ and count != 1):
            raise ValueError(f"Expected one visible GPU per Slurm task; found {count}; "
                             f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
        index = int(device.split(":")[1]) if ":" in device else 0
        if index >= count:
            raise ValueError(f"{device} is outside the task's {count} visible GPU(s); use cuda or cuda:0")


def run_item(args, name, config, identity):
    rank, size, affinity = identity
    out = args.root / name
    out.mkdir(exist_ok=True)
    with (out / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen, latest = out / "launch_config.yaml", out / "latest.pt"
        if frozen.exists():
            if yaml.safe_load(frozen.read_text()) != config:
                raise ValueError(f"{frozen}: differs from immutable root manifest")
            if not args.resume:
                raise ValueError(f"Existing experiment {out}; use --resume")
        else:
            if any(p.name != ".training.lock" for p in out.iterdir()):
                raise ValueError(f"{out}: missing original launch_config.yaml; refusing to overwrite history")
            atomic_yaml(frozen, config)
        if args.resume and completed(out, config):
            print(f"{name}: already completed; skipping", flush=True)
            return
        if (out / "completed.json").exists():
            raise ValueError(f"{out}: completion marker has no valid completed training result")
        resolve_collectors(deepcopy(config), len(affinity), frozen=True)
        if args.prepare_only:
            return
        validate_gpu(config)
        run_id = config.get("_launch", {}).get("run_id", f"{args.root.name}_{name}_s{config['experiment']['seed']}")
        command = [sys.executable, str(REPO / "run.py"), "--scenario", str(frozen),
                   "--output-dir", str(out), "--run-id", run_id, "--no-render"]
        if args.resume and latest.exists():
            command += ["--resume", str(latest)]
        elif args.resume:
            # Initialization can fail before the first recovery checkpoint. Keep
            # its artifacts before run.py opens fresh CSV/config output files.
            previous = out / f"uncheckpointed_{uuid.uuid4().hex}"
            artifacts = [p for p in out.iterdir()
                         if p.name not in {".training.lock", "launch_config.yaml", "train.log"}
                         and not p.name.startswith(("command_", "uncheckpointed_"))]
            if artifacts:
                previous.mkdir()
                for path in artifacts:
                    path.rename(previous / path.name)
            if artifacts or (out / "train.log").exists():
                print(f"{name}: no latest.pt; restarting from frozen initialization; "
                      f"prior train.log retained" + (f", other artifacts archived in {previous}" if artifacts else ""),
                      flush=True)
        environment = dict(os.environ)
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(key, None)
        attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{uuid.uuid4().hex[:8]}"
        record = dict(work_item=name, task=rank, tasks=size, cpu_affinity=affinity,
                      gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), node=os.environ.get("SLURMD_NODENAME"),
                      configuration=str(frozen), command=command,
                      source_code_sha256=source_code_sha256(),
                      allow_code_change=getattr(args, "allow_code_change", False))
        (out / f"command_{attempt}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Task {rank}/{size}: training {name}; output={out}", flush=True)
        with (out / "train.log").open("a") as log:
            log.write(f"\nAttempt {attempt}: {'resume' if args.resume else 'start'}; item={name}\n")
            log.flush()
            result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name}: exit {result.returncode}; see {out / 'train.log'}; use --resume")
        if not completed(out, config):
            raise RuntimeError(f"{name}: training budget or checkpoint selection incomplete; use --resume")
        (out / "completed.json").write_text(json.dumps(dict(checkpoint=str(out / "final.pt"), run_id=run_id)) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="Required for new runs; recovery reads the root manifest")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Source to freeze; defaults to the scenario checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-code-change", action="store_true",
                        help="Explicitly accept unproven training-code compatibility on recovery; preserves frozen settings")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", choices=("pilot", "full"))
    parser.add_argument("--steps-per-task", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--envs-per-worker", type=int)
    parser.add_argument("--reserved-cpus", type=int)
    parser.add_argument("--plan", action="store_true", help="Resolve and print assignment; no writes or training")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze configurations without training")
    parser.add_argument("--independent-cpu", action="store_true", help="Require one CPU-only independent task per node")
    parser.add_argument("--shared-token", help="Verify the run directory is shared with the batch host")
    parser.add_argument("--verify-allocation", action="store_true", help="Verify the assigned work item before training")
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    if not args.scenario and not args.resume:
        parser.error("New experiments require --scenario")
    if args.allow_code_change and not args.resume:
        parser.error("--allow-code-change requires --resume")
    if args.resume and (cli_overrides(args) or args.source):
        parser.error("Recovery uses frozen settings; omit source, profile, seed and configuration overrides")
    identity = task_identity(independent_cpu=True) if args.independent_cpu else task_identity()
    rank, size, affinity = identity
    if args.plan:
        if args.resume:
            data = yaml.safe_load((args.root / "launch_manifest.yaml").read_text())
            validate_revision(data, allow_code_change=args.allow_code_change)
            configs = data["items"]
            for config in configs.values():
                resolve_collectors(deepcopy(config), len(affinity), frozen=True)
        else:
            configs = prepare_configs(args, len(affinity))[0]
        print(yaml.safe_dump({"assignment": {r: list(configs)[r::size] for r in range(size)},
                             "collectors": {n: {k: c["experiment"][k] for k in ("parallel_envs", "envs_per_worker")}
                                            for n, c in configs.items()}}, sort_keys=False))
        return
    if args.independent_cpu and (not args.shared_token or
            (args.root / ".allocation.lock").read_text().strip() != args.shared_token):
        raise ValueError(f"{args.root} is not shared with the batch host")
    data = prepare_manifest(args, len(affinity))
    if args.independent_cpu:
        if len(data["items"]) != size:
            raise ValueError(f"Expected one independent work item per CPU node, got {len(data['items'])} items for {size} tasks")
        name = list(data["items"])[rank]
        config = data["items"][name]
        verify_cpu_item(config, affinity)
        if args.verify_allocation:
            from agents.common.checkpoints import read_checkpoint
            from core.provenance import collect_map_protocols

            maps = config["maps"]
            if not set(maps["selection"]).isdisjoint(maps["final_test"]):
                raise ValueError(f"{name}: checkpoint selection and final-test maps overlap")
            protocols = collect_map_protocols(config["environment"])
            missing = [bundle for bundle in maps["available"]
                       if bundle not in protocols or not protocols[bundle]["assets"]["image"]["sha256"]]
            if missing:
                raise ValueError(f"{name}: missing map files on this node: {', '.join(missing)}")
            source_id = data["source"]
            if config["adaptation"]["mode"] == "lora":
                source = read_checkpoint(source_id["path"])["metadata"]
                if source["algorithm"] != "ppo" or source["adaptation"]["lora"] is not None:
                    raise ValueError(f"{name}: expected a pretrained PPO base checkpoint")
            print(f"Verified {name}: rank={rank} node={os.environ.get('SLURMD_NODENAME', 'local')} "
                  f"cpus={os.environ.get('SLURM_CPUS_PER_TASK', len(affinity))} "
                  f"collectors={config['experiment']['parallel_envs']}x{config['experiment']['envs_per_worker']} "
                  f"eval_workers=1 cuda=0 source_sha256={source_id['sha256'] if source_id else 'none'} "
                  f"root={args.root}", flush=True)
            return
    elif args.verify_allocation:
        parser.error("--verify-allocation requires --independent-cpu")
    failures = []
    for name in list(data["items"])[rank::size]:
        try:
            run_item(args, name, data["items"][name], identity)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"{name}: {exc}", file=sys.stderr, flush=True)
            failures.append(name)
    if failures:
        raise SystemExit("Failed work items: " + ", ".join(failures))


if __name__ == "__main__":
    main()
