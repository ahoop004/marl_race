"""Build isolated checkpoint selection tasks without disturbing training RNGs."""
from copy import deepcopy
import fcntl
import json
from multiprocessing import get_context
import os
from pathlib import Path
import resource
import time

from adapters.rewards import RewardMapping
from training.evaluation_config import resolve_evaluation_protocol
from core.task_builder import create_race_task
from training.algorithms import check_evaluation_spec
from training.runtime import preserve_random_state, seed_process


def create_selection_evaluator(algorithm, scenario, scenario_dir, training_spec, params):
    protocol = resolve_evaluation_protocol(scenario, "selection")
    evaluation = deepcopy(scenario)
    evaluation["experiment"]["seed"] = protocol["seed"]
    evaluation["environment"]["max_steps"] = protocol["max_steps"]
    evaluation.setdefault("evaluation", {})["max_steps"] = protocol["max_steps"]
    if scenario["experiment"].get("async_evaluation") and "map_bundles" in protocol:
        evaluation["environment"]["map_bundles_eval"] = protocol["map_bundles"]
    if "target_laps" in protocol:
        evaluation["evaluation"]["target_laps"] = protocol["target_laps"]
    with preserve_random_state():
        seed_process(protocol["seed"])
        task = create_race_task(evaluation, scenario_dir=scenario_dir, mode="eval")
    try:
        check_evaluation_spec(training_spec, task.spec)
        options = dict(task=task, episodes=protocol["episodes"], base_seed=protocol["seed"],
                       reward_mapping=RewardMapping(params.get("reward_mode", "individual"),
                                                    params.get("team_reward_reduction", "mean")))
        if algorithm == "ppo":
            from training.ppo_evaluator import DeterministicPPOEvaluator
            return DeterministicPPOEvaluator(**options)
        if algorithm != "mappo":
            raise ValueError(f"Unsupported evaluation algorithm: {algorithm!r}")
        from training.mappo_evaluator import DeterministicMAPPOEvaluator
        return DeterministicMAPPOEvaluator(**options)
    except BaseException:
        task.close()
        raise


def verify_selected_checkpoint(output, config):
    from agents.common.checkpoints import checkpoint_sha256, read_checkpoint

    best = read_checkpoint(output / "best.pt")
    metadata = best["metadata"]
    selection = metadata["selection"]
    run_id = config.get("_launch", {}).get("run_id", f"{output.parent.name}_{output.name}_s{config['experiment']['seed']}")
    protocol = resolve_evaluation_protocol(config, "selection")
    seeds = list(range(protocol["seed"], protocol["seed"] + protocol["episodes"]))
    if (not selection or metadata["algorithm"] != "ppo" or best["training"] is not None
            or selection["run_id"] != run_id or metadata["provenance"]["run_id"] != run_id
            or selection["policy_version"] != metadata["progress"]["updates"]
            or selection["environment_steps"] != metadata["progress"]["environment_steps"]
            or selection["selection_strategy"] != config["evaluation"].get("selection_strategy", "completion_progress")
            or selection["evaluation_protocol"]["name"] != "selection"
            or selection["evaluation_protocol"]["max_steps"] != protocol["max_steps"]
            or selection["evaluation_protocol"]["target_laps"] != protocol["target_laps"]
            or selection["evaluation_protocol"]["seeds"] != seeds):
        raise ValueError("best.pt does not match its run and validation protocol")
    episodes = selection["episode_results"]
    maps = protocol.get("map_bundles", config["environment"].get("map_bundles_eval"))
    if [row["seed"] for row in episodes] != seeds or (maps and any(row["map_id"] not in maps for row in episodes)):
        raise ValueError("best.pt was not selected exclusively on the validation maps")
    if config["adaptation"]["mode"] == "lora":
        adaptation = metadata["adaptation"]
        if (adaptation["method"] != "lora" or adaptation["lora"]["rank"] != config["adaptation"]["lora"]["rank"]
                or adaptation["source_checkpoint"]["sha256"] != checkpoint_sha256(config["adaptation"]["checkpoint"])):
            raise ValueError("best.pt has the wrong LoRA configuration or pretrained source")
    with (output / "evaluation_history.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    winner = next((row for row in reversed(records) if row["is_best"]), None)
    if winner != selection:
        raise ValueError("best.pt differs from the recorded winning snapshot")
    latest = read_checkpoint(output / "latest.pt")["training"]
    pending = latest["hooks"]["EvaluationCheckpointHook"].get("pending_evaluations", ())
    completed = {(row["policy_version"], row.get("snapshot_sha256")) for row in records}
    if any((job["policy_version"], job["snapshot_sha256"]) not in completed for job in pending):
        raise ValueError("Checkpoint evaluation is still pending")


def _evaluate_selection_checkpoint(algorithm, scenario, scenario_dir, spec, params, checkpoint, result_path):
    """Run the selection protocol on a snapshot in a spawned process."""
    with checkpoint.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not result_path.is_file():
            _evaluate_and_save(algorithm, scenario, scenario_dir, spec, params, checkpoint, result_path)


def _evaluate_and_save(algorithm, scenario, scenario_dir, spec, params, checkpoint, result_path):
    import torch
    from agents.common.checkpoints import read_checkpoint
    from training.algorithms import create_learner

    torch.set_num_threads(1)
    evaluator = create_selection_evaluator(algorithm, scenario, scenario_dir, spec, params)
    try:
        policy = create_learner(algorithm, spec, params, training=False)
        snapshot = read_checkpoint(checkpoint)
        policy.load_for_evaluation(snapshot)
        started = time.perf_counter()
        started_unix = time.time()
        started_cpu = time.process_time()
        summary = evaluator.evaluate(policy, preserve_rng=False)
        seconds = time.perf_counter() - started
        summary.update(policy_inference_device=str(policy.device),
                       snapshot_sha256=snapshot["_identity"]["sha256"],
                       evaluation_started_unix=started_unix, evaluation_finished_unix=time.time(),
                       evaluation_cpu_seconds=time.process_time() - started_cpu,
                       evaluation_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = result_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump([summary, seconds], stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(result_path)
    finally:
        evaluator.close()


class BackgroundSelectionEvaluator:
    """Own the process and files for one checkpoint evaluation at a time."""

    def __init__(self, algorithm, scenario, scenario_dir, spec, params, workspace, *, device):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        policy_params = {**params, "device": device, "compile_networks": False}
        self._args = (algorithm, scenario, scenario_dir, spec, policy_params)
        self._process = None

    def snapshot(self, version):
        return self.workspace / f"{version:09d}.pt"

    def start(self, version):
        snapshot = self.snapshot(version)
        result = snapshot.with_suffix(".json")
        if result.is_file():
            return
        process = get_context("spawn").Process(target=_evaluate_selection_checkpoint,
                                               args=(*self._args, snapshot, result))
        process.start()
        self._process = process

    def result(self, version, *, wait=False):
        process = self._process
        if process is not None:
            if wait:
                process.join()
            status = process.exitcode
            if status is None:
                return None
            self.close()
            if status:
                raise RuntimeError(f"Evaluation of PPO update {version} exited with status {status}; resume to retry")
        return json.loads(self.snapshot(version).with_suffix(".json").read_text(encoding="utf-8"))

    def prune(self, versions):
        keep = {self.snapshot(version).stem for version in versions}
        for path in self.workspace.iterdir():
            if path.is_file() and path.stem not in keep:
                path.unlink()

    def close(self):
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        process.join()
        process.close()
        self._process = None
