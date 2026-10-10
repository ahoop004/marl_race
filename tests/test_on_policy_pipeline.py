"""Budget, trigger and sampling contracts of the shared native pipeline."""
import csv
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from tensordict import TensorDict

from test_torchrl_ppo import Capture, trainer, agent, rollout
from test_torchrl_mappo import learner, transition, IDS
from training.hooks import CSVHook, CheckpointHook, EvaluationCheckpointHook, WandbHook
from loggers.csv_logger import CSVLogger


@pytest.mark.parametrize("budget", [1, 3, 4, 6, 7, 9])
@pytest.mark.parametrize("frames_per_batch", [1, 2, 16])
def test_exact_decision_budgets_include_no_collector_padding(budget, frames_per_batch, monkeypatch):
    hook = Capture(True)
    training, controller = trainer(hook, frames_per_batch=frames_per_batch)
    batches = []
    update = training.agent.update

    def capture(data):
        batches.append(data.clone())
        return update(data)

    monkeypatch.setattr(training.agent, "update", capture)
    training.train(total_steps=budget)
    assert training.collected_steps == controller.calls == len(hook.steps) == budget
    assert training._physics_steps == training._agent_steps == budget
    assert sum(batch.numel() for batch in batches) == budget
    assert all((batch["collector", "traj_ids"] >= 0).all() for batch in batches)
    assert [row["train/rollout_steps"] for row in hook.updates] == (
        [4] * (budget // 4) + ([budget % 4] if budget % 4 else []))
    assert len(hook.episodes) == budget // 3
    assert controller.resets == budget // 3 + 1
    assert len(hook.starts) == math.ceil(budget / 3)
    assert sum(row["train/update_environment_steps"] for row in hook.updates) == budget
    assert sum(row["train/update_agent_samples"] for row in hook.updates) == budget
    assert hook.updates[-1]["train/completed_episodes"] == budget // 3
    assert hook.updates[-1]["train/optimizer_steps_total"] == sum(
        row["train/optimizer_steps"] for row in hook.updates)
    data = torch.cat(batches)
    assert data["next", "agents", "observation"].flatten().tolist() == [i % 3 + 1 for i in range(budget)]


@pytest.mark.parametrize("measure_services", [False, True])
def test_csv_wandb_evaluation_and_checkpoint_triggers_share_exact_counters(tmp_path, measure_services):
    hook = Capture(False)
    training, _ = trainer(hook)
    training.measure_services = measure_services
    messages = []
    logger = SimpleNamespace(log_metrics=messages.append, should_log=lambda group: True)
    evaluation_calls = []

    def evaluate():
        evaluation_calls.append(training.collected_steps)
        return dict(completion_rate=0., collision_rate=0., mean_net_progress=float(len(evaluation_calls)))

    evaluator = SimpleNamespace(evaluate=evaluate)
    training.hooks += [
        CSVHook(CSVLogger(str(tmp_path))), WandbHook(logger),
        CheckpointHook(training.agent, str(tmp_path), save_every_steps=4),
        EvaluationCheckpointHook(training.agent, str(tmp_path), evaluator, evaluate_every=1,
                                 evaluate_every_steps=4),
    ]
    training.train(total_steps=9)
    assert evaluation_calls == [4, 8]
    with (tmp_path / "update_metrics.csv").open() as stream:
        assert [int(row["train/environment_steps"]) for row in csv.DictReader(stream)] == [4, 8, 9]
    with (tmp_path / "episode_metrics.csv").open() as stream:
        assert [int(row["environment_steps"]) for row in csv.DictReader(stream)] == [3, 6, 9]
    assert [row["train/environment_steps"] for row in messages if "train/environment_steps" in row] == [4, 8, 9, 9]
    assert [row["episode/environment_steps"] for row in messages if "episode/environment_steps" in row] == [3, 6, 9]
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()
    final = torch.load(tmp_path / "final.pt", weights_only=True)
    assert final["metadata"]["progress"]["environment_steps"] == 9
    assert final["metadata"]["progress"]["updates"] == 3
    assert final["training"] is None
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert latest["training"]["progress"]["environment_steps"] == 9
    history = [json.loads(line) for line in (tmp_path / "evaluation_history.jsonl").read_text().splitlines()]
    assert [row["environment_steps"] for row in history] == [4, 8]
    assert history[-1]["is_best"]
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["environment_steps"] == summary["physics_steps"] == summary["agent_steps"] == 9
    assert summary["completed_episodes"] == 3 and summary["updates"] == 3
    service_metrics = training.service_metrics()
    if measure_services:
        assert set(service_metrics) == {"perf/logging_seconds", "perf/evaluation_seconds", "perf/checkpoint_seconds"}
        assert all(value > 0 for value in service_metrics.values())
        assert all(key in hook.updates[-1] for key in service_metrics)
    else:
        assert service_metrics == {} and not any(training.service_seconds.values())


def test_service_timers_exclude_nested_work_and_record_failures(monkeypatch):
    training, _ = trainer(Capture(False))
    training.measure_services = True
    ticks = iter([0., 1., 3., 5., 6., 9.])
    monkeypatch.setattr("training.on_policy.time.perf_counter", lambda: next(ticks))
    training.measure_service("logging", lambda: training.measure_service("evaluation", lambda: None))
    assert training.service_seconds == dict(logging=3., evaluation=2., checkpoint=0.)
    def fail():
        raise OSError("interrupted save")
    with pytest.raises(OSError, match="interrupted save"):
        training.measure_service("checkpoint", fail)
    assert training.service_seconds["checkpoint"] == 3.


def test_stage_timers_are_exclusive_and_optional(monkeypatch):
    training, _ = trainer(Capture(False))
    monkeypatch.setattr("training.on_policy.time.perf_counter", lambda: pytest.fail("disabled timer read the clock"))
    assert training.measure_stage("physics", lambda: 7) == 7
    assert training._stage_total() == 0
    training.measure_stages = True
    training._timing_phase = "warmup"
    ticks = iter([0., 1., 3., 5.])
    monkeypatch.setattr("training.on_policy.time.perf_counter", lambda: next(ticks))
    training.measure_stage("physics", lambda: training.measure_stage("lidar", lambda: None))
    assert training.stage_seconds["warmup"] == dict(physics=3., lidar=2.)
    assert training.stage_calls["warmup"] == dict(physics=1, lidar=1)


def test_stage_profiling_preserves_updates_and_restores_instance_methods():
    plain_capture, measured_capture = Capture(False), Capture(False)
    plain, _ = trainer(plain_capture)
    measured, _ = trainer(measured_capture)
    measured.measure_stages = True
    for module in ("actor", "critic"):
        getattr(measured.agent, module).load_state_dict(getattr(plain.agent, module).state_dict())
    rng = torch.random.get_rng_state()
    plain.train(total_steps=9)
    torch.random.set_rng_state(rng)
    measured.train(total_steps=9)
    assert plain.checkpoint_progress() == measured.checkpoint_progress()
    for module in ("actor", "critic"):
        for key, value in getattr(plain.agent, module).state_dict().items():
            torch.testing.assert_close(value, getattr(measured.agent, module).state_dict()[key], rtol=0, atol=0)
    stages = measured.stage_seconds
    for phase in ("warmup", "steady"):
        assert all(stages[phase][name] > 0 for name in ("collector", "environment_step", "policy_inference", "gae", "optimization", "tensor_transfer"))
    assert measured.stage_calls["warmup"]["optimization"] == 1
    assert measured.stage_calls["steady"]["optimization"] == 2
    assert "forward" not in vars(measured.agent.collection_policy)
    assert measured.agent.profile_call is None
    metrics = measured.performance_metrics()
    assert metrics["perf/initialization_seconds"] >= 0 and metrics["perf/warmup_seconds"] > 0
    assert all(metrics[f"perf/{name}_per_second"] > 0 for name in (
        "environment_steps", "physics_steps", "learner_samples", "optimizer_steps"))


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_kl_gate_preserves_parameters_before_first_optimizer_step(algorithm):
    policy = agent(target_kl=1e-6) if algorithm == "ppo" else learner(target_kl=1e-6)
    data = rollout(policy.policy, [0, 1, 2]) if algorithm == "ppo" else torch.stack(
        [transition(policy, IDS, step, 1, terminal=step == 2) for step in range(3)])
    with torch.no_grad():
        policy.actor.net[0].bias.add_(10)
    before = {key: value.clone() for key, value in policy.actor.state_dict().items()}
    metrics = policy.update(data)
    assert metrics["train/kl_early_stop"] == 1 and metrics["train/optimizer_steps"] == 0
    for key, value in policy.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_minibatch_sampler_visits_every_sample_once_including_short_tail():
    data = TensorDict({"sample_id": torch.arange(5)}, [5])
    replay = TensorDictReplayBuffer(storage=LazyTensorStorage(5),
                                   sampler=SamplerWithoutReplacement(drop_last=False), batch_size=2)
    replay.extend(data)
    for _ in range(3):
        samples = [replay.sample()["sample_id"] for _ in range(3)]
        assert [sample.numel() for sample in samples] == [2, 2, 1]
        assert torch.cat(samples).sort().values.tolist() == list(range(5))


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_disabling_kl_gate_preserves_losses_and_optimizer_behavior(algorithm):
    policy = agent() if algorithm == "ppo" else learner()
    checked = agent(target_kl=100) if algorithm == "ppo" else learner(target_kl=100)
    checked.actor.load_state_dict(policy.actor.state_dict())
    checked.critic.load_state_dict(policy.critic.state_dict())
    data = rollout(policy.policy, list(range(5))) if algorithm == "ppo" else torch.stack(
        [transition(policy, IDS, step, 1, terminal=step == 4) for step in range(5)])
    rng = torch.random.get_rng_state()
    metrics = policy.update(data.clone())
    torch.random.set_rng_state(rng)
    checked_metrics = checked.update(data.clone())
    assert metrics == checked_metrics
    for module in ("actor", "critic"):
        for name, value in getattr(policy, module).state_dict().items():
            torch.testing.assert_close(value, getattr(checked, module).state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_fixed_only_final_budget_reports_counters_without_a_fake_update(algorithm, tmp_path):
    from test_race_task import make_task
    from test_native_torchrl_env import team_task
    from training.on_policy import OnPolicyTrainer
    from training.hooks import ConsoleHook
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from test_torchrl_mappo import BOUNDS, params

    if algorithm == "ppo":
        task, _, _ = make_task({1: (("learner",), False, ("learner",), ()),
                                5: ((), True, (), ("opponent",))}, repeat=2)
        task.env.render_mode = None
        task.env.close = lambda: None
        policy = agent(n_steps=1)
        budget, physics, samples, updates = 3, 5, 1, 1
    else:
        task = team_task({1: (("car_0",), False, ("car_0",), ()),
                          3: (("car_1",), False, ("car_1",), ()),
                          5: ((), True, (), ("car_2", "car_3"))})
        policy = TorchRLMAPPOAgent(1, task.state_space().shape[0], *BOUNDS, IDS, params(n_steps=2))
        budget, physics, samples, updates = 4, 4, 4, 2
    capture, wandb, console = Capture(True), [], []
    training = OnPolicyTrainer(task, policy, hooks=[
        capture, CSVHook(CSVLogger(tmp_path)),
        WandbHook(SimpleNamespace(log_metrics=wandb.append)),
        ConsoleHook(SimpleNamespace(print_info=console.append)),
        CheckpointHook(policy, tmp_path),
    ])
    training.train(total_steps=budget)
    assert training.collected_steps == budget and training._physics_steps == physics
    assert training._agent_steps == len(capture.steps) == samples
    assert len(capture.updates) == updates
    assert wandb[-1] == training.counter_metrics()
    assert console[-1].startswith(f"Training stopped: environment_steps={budget}")
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["environment_steps"] == budget and summary["physics_steps"] == physics
    assert summary["agent_steps"] == samples and summary["updates"] == updates
    final = torch.load(tmp_path / "final.pt", weights_only=True)
    assert final["metadata"]["progress"] == training.checkpoint_progress()
    with (tmp_path / "update_metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == updates
