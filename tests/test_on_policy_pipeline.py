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
def test_exact_decision_budgets_include_no_collector_padding(budget, monkeypatch):
    hook = Capture(True)
    training, controller = trainer(hook)
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
    data = torch.cat(batches)
    assert data["next", "agents", "observation"].flatten().tolist() == [i % 3 + 1 for i in range(budget)]


def test_csv_wandb_evaluation_and_checkpoint_triggers_share_exact_counters(tmp_path):
    hook = Capture(False)
    training, _ = trainer(hook)
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
        assert len(list(csv.DictReader(stream))) == 3
    assert [row["train/environment_steps"] for row in messages if "train/environment_steps" in row] == [4, 8, 9]
    assert (tmp_path / "checkpoint_step000000004.pt").exists()
    assert (tmp_path / "checkpoint_step000000008.pt").exists()
    final = torch.load(tmp_path / "final_model.pt", weights_only=False)
    assert final["environment_steps"] == 9 and final["policy_version"] == 3
    history = [json.loads(line) for line in (tmp_path / "evaluation_history.jsonl").read_text().splitlines()]
    assert [row["environment_steps"] for row in history] == [4, 8]
    assert history[-1]["is_best"]


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
