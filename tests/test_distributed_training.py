"""Small real process-group tests; production launch remains CUDA/NCCL only."""
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training.distributed import DistributedExecution, distributed_execution


def _policy(mode, directory, **extra):
    from test_torchrl_ppo import agent
    from test_torchrl_mappo import learner
    from agents.common.ppo_policy import PPOPolicy
    from test_torchrl_mappo import BOUNDS
    options = dict(hidden_dims=[4], n_steps=1, n_epochs=2, batch_size=64, **extra)
    if mode == "ppo":
        return agent(**options)
    options["actor_mode"] = "independent" if mode == "independent" else "shared"
    if mode == "lora":
        options["lora"] = {"mode": "per_agent", "rank": 1, "alpha": 2, "per_agent_log_std": True}
    result = learner(**options)
    if mode == "lora":
        source = directory / f"source-{dist.get_rank()}.pt"
        PPOPolicy(1, *BOUNDS, {"hidden_dims": [4], "device": "cpu"}).save_checkpoint(source)
        result.initialize_from_checkpoint(source, scope="actor_only")
    return result


def _fragments(policy, mode, rank, update):
    from test_torchrl_ppo import rollout
    from test_torchrl_mappo import transition, IDS
    if rank == 1 and update == 0:
        return []
    if mode == "ppo":
        return [rollout(policy.policy, [float(rank + index) for index in range(3 - rank)])]
    ids = ["car_0"] if update == 2 else ([IDS[rank]] if update == 1 else IDS)
    return [torch.stack([transition(policy, ids, step, rank + step + 1, terminal=step == 2)
                         for step in range(3 - rank)])]


def _assert_models_match(policy, reference):
    for component in ("actor", "critic"):
        for key, value in getattr(policy, component).state_dict().items():
            torch.testing.assert_close(value, getattr(reference, component).state_dict()[key],
                                       atol=2e-6, rtol=2e-5, msg=lambda message: f"{component}.{key}: {message}")


def _numerical_worker(rank, size, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size,
                            timeout=timedelta(seconds=60))
    execution = DistributedExecution(rank, rank, size, torch.device("cpu"))
    try:
        expected = [(300, 300)] if size == 1 else [(256, 512), (44, 88)]
        assert list(execution.minibatches(300, 512)) == expected
        advantages = torch.arange(3 - rank, dtype=torch.float32) + rank * 4
        total = int(execution.sum(advantages.numel()))
        normalized = execution.normalize_advantage(advantages, total)
        combined = torch.cat(execution.gather(advantages))
        actual = torch.cat(execution.gather(normalized))
        torch.testing.assert_close(actual, (combined - combined.mean()) / combined.std(correction=0),
                                   atol=1e-7, rtol=1e-6)
        for mode in ("ppo", "shared", "independent", "lora"):
            torch.manual_seed(17)
            policy = _policy(mode, Path(directory))
            reference = _policy(mode, Path(directory))
            reference.actor.load_state_dict(policy.actor.state_dict())
            reference.critic.load_state_dict(policy.critic.state_dict())
            if rank:
                with torch.no_grad():
                    for p in policy.actor.parameters():
                        p.add_(1)
            execution.attach(policy)
            _assert_models_match(policy, reference)
            frozen = {name: p.clone() for name, p in policy.actor.named_parameters() if not p.requires_grad}
            for update in range(3):
                fragments = _fragments(policy, mode, rank, update)
                all_fragments = [fragment for packet in execution.gather(fragments) for fragment in packet]
                actual = policy.update_rollouts(fragments)
                expected = reference.update_rollouts(all_fragments)
                assert actual["train/optimizer_steps"] == expected["train/optimizer_steps"] == 2
                _assert_models_match(policy, reference)
                for name, p in policy.actor.named_parameters():
                    if name in frozen:
                        torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
            # A global KL gate must stop EVERY rank before optimizer work.
            policy.target_kl = 1e-5
            fragments = _fragments(policy, mode, rank, 1)
            raw_key = "raw_log_prob" if mode == "ppo" else ("agents", "raw_log_prob")
            for fragment in fragments:
                fragment[raw_key].add_(2 if rank == 0 else 0)
            before = [p.clone() for p in policy._optim_parameters]
            metrics = policy.update_rollouts(fragments)
            assert metrics["train/kl_early_stop"] == 1 and metrics["train/optimizer_steps"] == 0
            for p, old in zip(policy._optim_parameters, before):
                torch.testing.assert_close(p, old, atol=0, rtol=0)
            # Three global minibatches per epoch, including a one-sample tail
            # with an empty rank. Identical rows make sampler order irrelevant.
            from copy import deepcopy
            from agents.torchrl_updates import optimize_ppo
            from tensordict import TensorDict
            policy.target_kl = reference.target_kl = None
            policy.batch_size = reference.batch_size = 3
            reference.optimizer.load_state_dict(deepcopy(policy.optimizer.state_dict()))
            row = _fragments(policy, mode, 0, 2)[0][:1]
            if mode != "ppo":
                row = TensorDict({"agents": row["agents"][:, :1], "state": row["state"]}, [1])
            for name in ("advantage", "value_target", "state_value"):
                field = name if mode == "ppo" else ("agents", name)
                shape = (1, 1) if mode == "ppo" else (1, 1, 1)
                row[field] = torch.ones(shape)
            row = execution.broadcast(row)
            local_count = (7 if size == 1 else 5 - 3 * rank)
            actual = optimize_ppo(policy, row.expand(local_count).clone(),
                                  group=None if mode == "ppo" else "agents")
            expected = optimize_ppo(reference, row.expand(7).clone(),
                                    group=None if mode == "ppo" else "agents")
            assert actual["train/optimizer_steps"] == expected["train/optimizer_steps"] == 6
            _assert_models_match(policy, reference)
            execution.detach(policy)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [1, 2])
def test_ddp_matches_global_batch_for_ppo_and_all_mappo_actor_modes(size, tmp_path):
    mp.spawn(_numerical_worker, args=(size, f"file://{tmp_path / 'group'}", str(tmp_path)),
             nprocs=size, join=True)


def test_launcher_validation_rejects_missing_launcher_and_unavailable_gpus(monkeypatch):
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="torchrun"):
        with distributed_execution(True):
            pass

    for key, value in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                           MASTER_ADDR="localhost", MASTER_PORT="29877").items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="experiment.distributed=true"):
        with distributed_execution(False):
            pass
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="visible CUDA"):
        with distributed_execution(True):
            pass
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="single-node"):
        with distributed_execution(True):
            pass


def test_process_group_is_destroyed_after_a_training_failure(monkeypatch):
    for key, value in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
                           MASTER_ADDR="localhost", MASTER_PORT="29877").items():
        monkeypatch.setenv(key, value)
    events = []
    initialized = [False]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: events.append(("device", index)))
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: initialized[0])
    def initialize(*args, **kwargs):
        initialized[0] = True
        events.append(("initialize", kwargs["device_id"]))
    def destroy():
        initialized[0] = False
        events.append(("destroy", None))
    monkeypatch.setattr(dist, "init_process_group", initialize)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="controlled"):
        with distributed_execution(True) as execution:
            assert execution.device == torch.device("cuda:0")
            raise RuntimeError("controlled training failure")
    assert [event[0] for event in events] == ["device", "initialize", "destroy"]
    assert not initialized[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rank_local_cuda_rng_round_trip():
    from agents.common.checkpoints import capture_rng, restore_rng
    from training.runtime import preserve_random_state
    device = torch.device("cuda", torch.cuda.current_device())
    with preserve_random_state():
        state = capture_rng(device)
        assert state["cuda"] == [] and state["cuda_local"]["device"] == device.index
        expected_cpu, expected_cuda = torch.rand(4), torch.rand(4, device=device)
        restore_rng(state)
        torch.testing.assert_close(torch.rand(4), expected_cpu, atol=0, rtol=0)
        torch.testing.assert_close(torch.rand(4, device=device), expected_cuda, atol=0, rtol=0)


def _collector_worker(rank, rendezvous, directory, backend):
    from contextlib import ExitStack
    from types import SimpleNamespace
    import json
    from core.configuration import PROJECT_ROOT
    from core.scenario import load_and_expand_scenario
    from core.agent_roles import resolve_agent_roles
    from application.train import _run_training
    from loggers.console import ConsoleLogger
    from agents.common.checkpoints import read_checkpoint
    from training.algorithms import learner_params, create_learner
    from core.task_builder import create_race_task

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    execution = DistributedExecution(rank, rank, 2, torch.device("cpu"))
    try:
        for algorithm in ("ppo", "mappo"):
            name = "ppo_lap_completion_pretrain.yaml" if algorithm == "ppo" else "mappo_2v2_completion_scratch.yaml"
            scenario_path = PROJECT_ROOT / "scenarios" / name
            scenario = load_and_expand_scenario(scenario_path, overrides=[
                "experiment.distributed=true", "experiment.parallel_envs=2", "experiment.envs_per_worker=2",
                f"++experiment.collection_backend={backend}", "experiment.total_steps=16",
                "++experiment.stop_after_steps=8", "environment.max_steps=2", "evaluation.enabled=false",
                "wandb.enabled=false", "execution.params.device=cpu", "algorithm.params.n_steps=8",
                "algorithm.params.n_epochs=1", "algorithm.params.batch_size=4",
                "++algorithm.params.frames_per_batch=4", "network.actor_hidden_dims=[4]",
                "network.critic_hidden_dims=[4]", "checkpoint.every_updates=1"])
            output = Path(directory) / algorithm
            args = SimpleNamespace(checkpoint=None, run_id=f"ddp-{algorithm}", output_dir=str(output),
                                   scenario=str(scenario_path), pretrained_actor=None)
            roles = resolve_agent_roles(scenario["agents"])
            with ExitStack() as resources:
                _run_training(scenario, args, ConsoleLogger(False), scenario_path.parent, roles,
                              resources, execution)
            state = read_checkpoint(output / "latest.pt")
            assert state["training"]["progress"]["environment_steps"] == 8
            assert len(state["training"]["distributed"]["ranks"]) == 2
            assert not (output / "final.pt").exists()
            scenario["experiment"].pop("stop_after_steps")
            scenario["checkpoint"]["resume"] = str(output / "latest.pt")
            scenario["experiment"]["parallel_envs"] = 4
            with ExitStack() as resources, pytest.raises(ValueError, match="collector layout"):
                _run_training(scenario, args, ConsoleLogger(False), scenario_path.parent, roles,
                              resources, execution)
            scenario["experiment"]["parallel_envs"] = 2
            with ExitStack() as resources:
                _run_training(scenario, args, ConsoleLogger(False), scenario_path.parent, roles,
                              resources, execution)
            state = read_checkpoint(output / "latest.pt")
            assert state["training"]["progress"]["environment_steps"] == 16
            assert state["training"]["progress"]["updates"] >= 2
            assert all(not key.startswith("module.") for key in state["models"]["actor"])
            # Evaluation construction is ordinary and independent of training world size.
            task = create_race_task(scenario, scenario_dir=scenario_path.parent, roles=roles)
            try:
                evaluator = create_learner(algorithm, task.spec, learner_params(scenario, task.spec, algorithm), training=False)
                evaluator.load_for_evaluation(output / "final.pt")
                for key, value in evaluator.actor.state_dict().items():
                    torch.testing.assert_close(value, state["models"]["actor"][key], atol=0, rtol=0)
            finally:
                task.close()
            summary = json.loads((output / "run_summary.json").read_text())
            assert summary["environment_steps"] == 16
            assert summary["agent_steps"] == 16 * (1 if algorithm == "ppo" else 2)
            # Every completed race from both ranks is written exactly once by rank zero.
            records = [json.loads(line) for line in (output / "race_metrics.jsonl").read_text().splitlines()]
            assert len({record["episode_id"] for record in records}) == len(records)
            assert {record["environment_id"] for record in records} == {0, 1, 2, 3}
            assert {record["environment_seed"] for record in records} == {42, 43, 44, 45}
        assert not mp.active_children()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["centralized", "process"])
def test_two_ranks_collect_update_checkpoint_and_resume_both_algorithms(tmp_path, backend):
    mp.spawn(_collector_worker, args=(f"file://{tmp_path / 'group'}", str(tmp_path), backend), nprocs=2, join=True)
