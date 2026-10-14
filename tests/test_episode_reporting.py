"""Reporting order and identity contracts, without starting collector workers."""
import csv
import json
import sys
from types import SimpleNamespace

import pytest

from loggers.csv_logger import CSVLogger
from loggers.metric_policy import MetricPolicy
from loggers.wandb_logger import WandbLogger
from training.hooks import ConsoleHook, CSVHook, WandbHook
from training.race_reporting import RaceTrainingReports
from test_torchrl_ppo import Capture, trainer


def queued_episode(episode, steps, *, race=True):
    identity = dict(episode_id=f"run_ep{episode:06d}", environment_id=episode % 512,
                    environment_episode=episode // 512)
    metrics = {**identity, "episode_steps": 3, "lap_time_s": 154.,
               "train/environment_steps": steps,
               "agent_rewards": {"car_0": 3.9},
               "agent_outcomes": {"car_0": "finished"},
               "agent_terminal_reasons": {"car_0": "target_laps"}}
    if race:
        metrics["race_record"] = {
            **identity, "episode": episode, "reported_at_environment_steps": steps,
            "mean_net_progress_laps": 1., "spawn_ids": {"car_0": "spawn"},
            "agents": {"car_0": {"team": "trainable"}},
        }
    return episode, 3.9, {"outcome": "finished", "lap_count": 1}, metrics


def reports_for(hooks, *, completed=0):
    owner = SimpleNamespace(task=SimpleNamespace(possible_agents=["car_0"]),
                            agent=SimpleNamespace(), hooks=hooks,
                            call_hook=lambda hook, method, *args: getattr(hook, method)(*args))
    reports = RaceTrainingReports(owner)
    reports.completed = completed
    return reports


def dispatch(reports, episodes, *, parent=True):
    reports.pending_episodes.extend(episodes)
    reports.completed += len(episodes)
    reports.dispatch({"train/environment_steps": 999999}, parent=parent)


@pytest.mark.parametrize("race", [False, True])
def test_interleaved_ids_keep_identity_and_one_record_per_completion(tmp_path, monkeypatch, race):
    records, definitions, log_steps, messages = [], [], [], []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: SimpleNamespace(id="existing-run", name="test", url="test"),
        define_metric=lambda *args, **kwargs: definitions.append((args, kwargs)),
        log=lambda row, **kwargs: (records.append(row), log_steps.append(kwargs)),
    ))
    wandb = WandbLogger("test", mode="online")
    csv_logger = CSVLogger(tmp_path, scenario_config={"logging": {"csv_exports": True}})
    capture = Capture(False)
    reports = reports_for([ConsoleHook(SimpleNamespace(print_info=messages.append)),
                           WandbHook(wandb), CSVHook(csv_logger), capture])
    ids = [352, 495, 17, 56, 864, 513]
    # Multiple dispatches must not reset numbering. One slot completes twice.
    dispatch(reports, [queued_episode(ep, 100 + i, race=race) for i, ep in enumerate(ids[:4])])
    dispatch(reports, [queued_episode(ep, 104 + i, race=race) for i, ep in enumerate(ids[4:])])
    reports.dispatch({})  # Empty dispatches produce no records or indices.
    csv_logger.close()

    assert reports.completed == 6
    assert [row[0] for row in capture.episodes] == ids
    assert [row[3]["episode/completion_index"] for row in capture.episodes] == list(range(6))
    assert [int(message.split()[1]) for message in messages if message.startswith("ep ")] == list(range(6))
    assert "env=352  reward=+3.90" in messages[0]
    assert [row["episode/number"] for row in records] == ids
    assert [row["episode/completion_index"] for row in records] == list(range(6))
    assert [row["episode/environment_steps"] for row in records] == list(range(100, 106))
    assert [row["episode/environment_id"] for row in records] == [ep % 512 for ep in ids]
    assert [row["episode/environment_episode"] for row in records] == [ep // 512 for ep in ids]
    assert all(row["episode/reward"] == 3.9 and row["episode/steps"] == 3
               and row["episode/completed"] == 1 and row["episode/failed"] == 0
               and row["episode/timeout"] == 0 for row in records)
    if race:
        assert all(row["episode/net_progress_laps"] == 1. for row in records)
    assert log_steps == [{"step": None}] * 6  # Let W&B manage its history step.
    assert (("episode/*",), {"step_metric": "episode/environment_steps"}) in definitions
    assert (("episode/completion_index",), {}) in definitions

    with (tmp_path / "episode_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["episode"]) for row in rows] == ids
    assert [int(row["completion_index"]) for row in rows] == list(range(6))
    assert [int(row["environment_steps"]) for row in rows] == list(range(100, 106))
    assert [int(row["environment_id"]) for row in rows] == [ep % 512 for ep in ids]
    assert [int(row["environment_episode"]) for row in rows] == [ep // 512 for ep in ids]
    assert [row["episode_id"] for row in rows] == [f"run_ep{ep:06d}" for ep in ids]
    with (tmp_path / "agent_metrics.csv").open() as stream:
        agent_rows = list(csv.DictReader(stream))
    assert [int(row["completion_index"]) for row in agent_rows] == list(range(6))
    assert [int(row["reported_at_environment_steps"]) for row in agent_rows] == list(range(100, 106))
    if race:
        races = [json.loads(line) for line in (tmp_path / "race_metrics.jsonl").read_text().splitlines()]
        assert [row["completion_index"] for row in races] == list(range(6))
        assert [row["episode_id"] for row in races] == [f"run_ep{ep:06d}" for ep in ids]


@pytest.mark.parametrize("completed", [0, 24, 9826])
@pytest.mark.parametrize("lap_completion", [False, True])
def test_summary_cadence_uses_global_completions_across_resume(completed, lap_completion):
    messages = []
    hook = ConsoleHook(SimpleNamespace(print_info=messages.append), summary_every=25,
                       lap_completion=lap_completion)
    reports = reports_for([hook], completed=completed)
    summary_indices = []
    for offset in range(50):
        # Every environment-specific ID is divisible by 25: unsuitable cadence.
        episode = queued_episode(offset * 512 * 25, offset + 1)
        episode[3]["agent_rewards"]["car_1"] = 1.
        episode[3]["agent_outcomes"]["car_1"] = "timeout"
        episode[3]["agent_terminal_reasons"]["car_1"] = "time_limit"
        before = len(messages)
        dispatch(reports, [episode])
        if any("outcomes (last" in message for message in messages[before:]):
            summary_indices.append(completed + offset)
    assert summary_indices == [index for index in range(completed, completed + 50)
                               if (index + 1) % 25 == 0]
    assert len(summary_indices) == 2
    if lap_completion:
        assert any("learner outcomes" in message and "timeout" in message for message in messages)
    else:
        assert any("car_1 outcomes" in message for message in messages)


def test_workers_forward_original_identity_without_assigning_global_index():
    capture = Capture(False)
    reports = reports_for([capture], completed=100)
    dispatch(reports, [queued_episode(352, 1), queued_episode(864, 2)], parent=False)
    assert [row[0] for row in capture.episodes] == [352, 864]
    assert all("episode/completion_index" not in row[3]
               and "completion_index" not in row[3]["race_record"] for row in capture.episodes)


def test_serial_environment_callbacks_report_sequential_completions_without_collection():
    from adapters.native_torchrl import NativeRaceTorchRLEnv
    from test_native_torchrl_env import zero_policy

    capture = Capture(False)
    training, _ = trainer(capture)
    env = training.env = NativeRaceTorchRLEnv(training.task)
    env.on_reset, env.on_physics_step, env.on_decision = (
        training.reports.reset, training.reports.physics, training.reports.decision)
    try:
        for _ in range(2):
            current = env.reset()
            for _ in range(3):
                result = env.step(zero_policy(current))
                current = result["next"]
            training.reports.dispatch({})
        assert [row[0] for row in capture.episodes] == [0, 1]
        assert [row[3]["episode/completion_index"] for row in capture.episodes] == [0, 1]
        assert [row[3]["environment_id"] for row in capture.episodes] == [0, 0]
        assert [row[3]["environment_episode"] for row in capture.episodes] == [0, 1]
        assert [row[3]["episode_id"] for row in capture.episodes] == ["run_ep000000", "run_ep000001"]
        assert [row[3]["train/environment_steps"] for row in capture.episodes] == [3, 6]
        assert training.reports.completed == 2 and training.reports.next_episode == 2
    finally:
        env.close()


def test_csv_append_expands_old_schema_and_keeps_original_rows(tmp_path):
    path = tmp_path / "episode_metrics.csv"
    path.write_text("episode,reward,environment_steps\n352,3.9,100\n")
    logger = CSVLogger(tmp_path, append=True)
    reports = reports_for([CSVHook(logger)], completed=9826)
    dispatch(reports, [queued_episode(864, 200, race=False)])
    logger.close()
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames[:3] == ["episode", "reward", "environment_steps"]
        old, new = list(reader)
    assert old["episode"] == "352" and old["reward"] == "3.9"
    assert old["completion_index"] == ""
    assert new["episode"] == "864" and new["completion_index"] == "9826"


def test_checkpoint_without_completion_index_restores_reporting_progress(tmp_path):
    from agents.common.checkpoints import read_checkpoint

    original, _ = trainer(Capture(False))
    original._environment_steps, original._physics_steps, original._agent_steps = (12288000,) * 3
    original._updates, original._optimizer_steps = 30, 120000
    original.reports.completed, original.reports.next_episode = 9826, 11122
    original.total_steps, original.checkpoint_ready = 120000000, True
    state = original.checkpoint_state()
    assert "completion_index" not in state["progress"]
    path = tmp_path / "latest.pt"
    original.agent.save_checkpoint(path, training_state=state,
                                   provenance={"run_id": "run", "wandb_run_id": "existing-run"})
    saved = read_checkpoint(path)
    assert saved["schema_version"] == 1
    assert saved["metadata"]["provenance"]["wandb_run_id"] == "existing-run"
    capture = Capture(False)
    recovered, _ = trainer(capture)
    restored, _ = recovered.agent.resume_training(path)
    recovered.restore_training_state(restored)
    assert recovered.checkpoint_progress() == original.checkpoint_progress()
    assert recovered.reports.next_episode == 11122
    dispatch(recovered.reports, [queued_episode(11122, 12288001), queued_episode(11634, 12288002)])
    assert [row[3]["episode/completion_index"] for row in capture.episodes] == [9826, 9827]
    assert recovered.reports.completed == 9828
    assert [row[0] for row in capture.episodes] == [11122, 11634]
    assert recovered.agent.optimizer.param_groups[0]["lr"] == original.agent.optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("config", [{}, {"profile": "lap_completion"}, {"profile": "debug"},
                                    {"metrics": {"episode/reward": True}}])
def test_metric_profiles_keep_new_default_axis_with_episode_values(config):
    policy = MetricPolicy(config)
    fields = {"episode/reward": 3.9, "episode/number": 864,
              "episode/completion_index": 4, "episode/environment_id": 352,
              "episode/environment_episode": 1, "episode/environment_steps": 100}
    selected = policy.filter(fields)
    assert selected["episode/environment_steps"] == 100
    assert selected["episode/reward"] == 3.9
    if "metrics" not in config:
        assert selected == fields
    assert MetricPolicy({"groups": {"train": False}}).filter(fields) == {}
