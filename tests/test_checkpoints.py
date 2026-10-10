"""Versioned checkpoint safety, compatibility and update-boundary recovery."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from agents.common.checkpoints import read_checkpoint, restore_rng
from agents.common.ppo_policy import PPOPolicy
from application.checkpoints import resolve_checkpoint_path, validate_resume_experiment
from training.hooks import CheckpointHook
from test_torchrl_ppo import Capture, agent, trainer, rollout
from utils.torch_io import atomic_save, safe_load


@pytest.mark.parametrize("field", ["schema", "implementation", "network", "observation", "physics", "routing", "tensor"])
def test_incompatible_checkpoints_reject_before_model_mutation(field, tmp_path):
    source = agent()
    path = tmp_path / "model.pt"
    source.save_checkpoint(path)
    payload = read_checkpoint(path)
    if field == "schema":
        payload["schema_version"] = 99
    elif field == "implementation":
        payload["metadata"]["implementation"] = "unsupported_policy.v99"
    elif field == "network":
        payload["metadata"]["network"]["activation"] = "relu"
    elif field == "observation":
        payload["metadata"]["contracts"]["observations"]["policy"]["contract"] = {"order": ["changed"]}
    elif field == "physics":
        payload["metadata"]["contracts"]["physics"] = {"timestep": 123.}
    elif field == "routing":
        payload["metadata"]["structure"]["routing"]["policy"] = 1
    else:
        payload["models"]["actor"]["net.0.weight"] = torch.zeros(3, 3)
    atomic_save(payload, path)
    destination = agent()
    before = {name: value.clone() for name, value in destination.actor.state_dict().items()}
    with pytest.raises(ValueError, match="Incompatible|Unsupported"):
        destination.load_for_evaluation(path)
    for name, value in destination.actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_atomic_failure_preserves_valid_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    atomic_save({"value": torch.tensor(1)}, path)
    previous = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_save({"value": torch.tensor(2)}, path)
    assert path.read_bytes() == previous
    assert [item.name for item in tmp_path.iterdir()] == ["latest.pt"]


def test_historical_and_arbitrary_object_checkpoints_have_no_loading_fallback(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"actor": {"weight": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="historical"):
        read_checkpoint(path)
    torch.save({"object": np.ones(1)}, path)
    with pytest.raises(Exception, match="Weights only load failed"):
        safe_load(path)


def test_application_recovery_rejects_changed_map_yaml_hashes():
    from core.provenance import collect_map_protocols
    from core.scenario import load_and_expand_scenario

    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    metadata = {"configuration": scenario, "provenance": {
        "map_protocols": collect_map_protocols(scenario["environment"])}}
    validate_resume_experiment(metadata, scenario)
    first = next(iter(metadata["provenance"]["map_protocols"].values()))
    first["yaml_sha256"] = "changed_map_contents"
    with pytest.raises(ValueError, match="map_protocols"):
        validate_resume_experiment(metadata, scenario)


def test_explicit_ppo_transfer_restores_models_with_fresh_optimizer_and_schedule(tmp_path):
    source = agent(lr_schedule="linear", learning_rate_end=1e-4)
    source.set_training_progress(.7)
    source.update(rollout(source.policy, [0, 1, 2]))
    path = tmp_path / "ppo.pt"
    source.save_checkpoint(path)
    destination = agent(lr_schedule="linear", learning_rate_end=1e-4)
    destination.initialize_from_checkpoint(path, scope="actor_and_critic")
    assert not destination.optimizer.state
    assert destination.optimizer.param_groups[0]["lr"] == destination.lr
    for name in ("actor", "critic"):
        for key, value in getattr(source, name).state_dict().items():
            torch.testing.assert_close(value, getattr(destination, name).state_dict()[key])
    with pytest.raises(ValueError, match="model-only"):
        destination.resume_training(path)
    assert destination.source_checkpoint["scope"] == "actor_and_critic"
    payload = read_checkpoint(path)
    payload["metadata"]["implementation"] = "unsupported_policy.v99"
    atomic_save(payload, path)
    with pytest.raises(ValueError, match="implementation"):
        agent().initialize_from_checkpoint(path, scope="actor_only")


def test_latest_restores_optimizer_progress_rng_and_recovers_a_fresh_episode(tmp_path):
    class Stop(Capture):
        def on_update(self, metrics):
            super().on_update(metrics)
            self.should_stop = True

    hook = Stop(False)
    original, controller = trainer(hook, lr_schedule="linear", learning_rate_end=1e-4)
    original.hooks.append(CheckpointHook(original.agent, str(tmp_path), save_every_steps=4))
    original.hooks[-1].bind_trainer(original)
    original.train(total_steps=9)
    assert original.collected_steps == 4 and controller.calls == 4
    assert (tmp_path / "latest.pt").exists() and not (tmp_path / "final.pt").exists()
    saved = read_checkpoint(tmp_path / "latest.pt")
    assert saved["training"]["progress"]["completed_episodes"] == 1
    assert saved["training"]["recovery"]["discard_partial_episode"]
    assert saved["training"]["recovery"]["environment_schedules"] == "restart_at_seed"
    assert resolve_checkpoint_path(str(tmp_path), operation="resume").name == "latest.pt"

    recoveries = []
    for _ in range(2):
        capture = Capture(True)
        recovered, fixed = trainer(capture, lr_schedule="linear", learning_rate_end=1e-4)
        latest = CheckpointHook(recovered.agent, str(tmp_path / "recovered"), save_every_steps=4)
        recovered.hooks.append(latest)
        latest.bind_trainer(recovered)
        state, _ = recovered.agent.resume_training(tmp_path / "latest.pt")
        recovered.restore_training_state(state)
        assert recovered.agent.optimizer.state
        assert recovered.agent.optimizer.param_groups[0]["lr"] == pytest.approx(
            original.agent.optimizer.param_groups[0]["lr"])
        for key, value in original.agent.actor.state_dict().items():
            torch.testing.assert_close(value, recovered.agent.actor.state_dict()[key])
        restore_rng(state["rng"])
        expected = (random.random(), np.random.random(), torch.rand(2))
        restore_rng(state["rng"])
        actual = (random.random(), np.random.random(), torch.rand(2))
        assert expected[:2] == actual[:2]
        torch.testing.assert_close(expected[2], actual[2])
        recovered.train(total_steps=9)
        assert recovered.collected_steps == 9 and fixed.calls == len(capture.steps) == 5
        assert capture.steps[0].step_idx == 0
        assert [row["train/environment_steps"] for row in capture.updates] == [8, 9]
        assert recovered.agent.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        recoveries.append((capture.steps, deepcopy(recovered.agent.actor.state_dict())))
    for a, b in zip(recoveries[0][0], recoveries[1][0]):
        np.testing.assert_array_equal(a.action_norm, b.action_norm)
    for key, value in recoveries[0][1].items():
        torch.testing.assert_close(value, recoveries[1][1][key], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["shared", "independent", "lora"])
def test_mappo_recovery_restores_actor_ownership_and_lora_base(mode, tmp_path):
    from test_native_torchrl_env import team_task
    from test_torchrl_mappo import IDS, BOUNDS, params
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from training.on_policy import OnPolicyTrainer

    class Stop(Capture):
        def on_update(self, metrics):
            super().on_update(metrics)
            self.should_stop = True

    events = {3: ((), True, (), tuple([*IDS, "car_2", "car_3"]))}
    options = params(hidden_dims=[4], n_steps=2, actor_mode="shared" if mode == "lora" else mode)
    source_path = tmp_path / "source.pt"
    if mode == "lora":
        options["lora"] = {"mode": "per_agent", "rank": 1, "alpha": 2, "train_log_std": False,
                           "per_agent_log_std": True}
        PPOPolicy(1, *BOUNDS, {"hidden_dims": [4], "device": "cpu"}).save_checkpoint(source_path)
    task = team_task(events)
    original = TorchRLMAPPOAgent(1, task.state_space().shape[0], *BOUNDS, IDS, options)
    if mode == "lora":
        original.initialize_from_checkpoint(source_path, scope="actor_only")
    directory = tmp_path / "run"
    training = OnPolicyTrainer(task, original, hooks=[Stop(False), CheckpointHook(original, directory, save_every_steps=2)])
    training.train(total_steps=5)
    new_task = team_task(events)
    recovered = TorchRLMAPPOAgent(1, new_task.state_space().shape[0], *BOUNDS, IDS, options)
    state, metadata = recovered.resume_training(directory / "latest.pt")
    expected, _ = original.act_batch(IDS, np.zeros((2, 1), dtype=np.float32), deterministic=True)
    actual, _ = recovered.act_batch(IDS, np.zeros((2, 1), dtype=np.float32), deterministic=True)
    for aid in IDS:
        np.testing.assert_array_equal(expected[aid], actual[aid])
    if mode == "lora":
        assert all(not value.requires_grad for value in recovered.actor.net.parameters())
        assert all(not value.requires_grad for value in recovered.actor.log_stds)
        assert recovered.source_checkpoint["sha256"] == original.source_checkpoint["sha256"]
        assert metadata["adaptation"]["lora"]["agent_to_adapter"] == {"car_0": 0, "car_1": 1}
    capture = Capture(True)
    continued = OnPolicyTrainer(new_task, recovered, hooks=[capture, CheckpointHook(recovered, tmp_path / "recovery")])
    continued.restore_training_state(state)
    continued.train(total_steps=5)
    assert continued.collected_steps == 5 and len(capture.steps) == 6
    assert continued._agent_steps == 10


def test_observation_expansion_rejects_a_changed_driving_prefix(tmp_path):
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from wrappers.observations.composer import ObservationComposer
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from test_torchrl_mappo import BOUNDS, IDS, params

    directory = Path(__file__).resolve().parents[1] / "scenarios"
    contracts = [ObservationComposer.from_config(config["agents"]["car_0"]["observation"], config["environment"]).contract
                 for config in [load_and_expand_scenario(str(directory / f"{name}.yaml")) for name in
                                ("ppo_lap_completion_pretrain", "mappo_2v2_completion_scratch")]]
    source = PPOPolicy(158, *BOUNDS, {"hidden_dims": [4], "device": "cpu", "_observation_contract": contracts[0]})
    source_path = tmp_path / "ppo.pt"
    source.save_checkpoint(source_path)
    contract = deepcopy(contracts[1])
    contract["observation"]["lidar"]["normalize"] = not contract["observation"]["lidar"]["normalize"]
    target = TorchRLMAPPOAgent(192, 3, *BOUNDS, IDS, params(hidden_dims=[4], _observation_contract=contract))
    before = deepcopy(target.actor.state_dict())
    with pytest.raises(ValueError, match="observation_prefix"):
        target.initialize_from_checkpoint(source_path, scope="actor_only", observation_extension="frenet_neighbors")
    for key, value in before.items():
        torch.testing.assert_close(value, target.actor.state_dict()[key])
