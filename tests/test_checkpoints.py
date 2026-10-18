"""Versioned checkpoint safety, compatibility and update-boundary recovery."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from agents.common.checkpoints import read_checkpoint, restore_rng
from agents.common.ppo_policy import PPOPolicy
from application.checkpoints import resolve_checkpoint_path, validate_resume_experiment
from training.hooks import CheckpointHook, EvaluationCheckpointHook
from training.on_policy import OnPolicyTrainer
from test_torchrl_ppo import Capture, agent, trainer, rollout
from utils.torch_io import atomic_save, safe_load


def test_checkpoint_directories_select_best_for_transfer_and_latest_for_resume(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("best.pt", "latest.pt", "final.pt"):
        (source / name).write_bytes(b"path resolution only")
    monkeypatch.chdir(tmp_path.parent)
    for operation in ("evaluation", "transfer", "resume"):
        expected = "latest.pt" if operation == "resume" else "best.pt"
        assert resolve_checkpoint_path("source", operation=operation, base_dir=tmp_path) == source / expected
    assert resolve_checkpoint_path(str(source / "final.pt"), operation="transfer") == source / "final.pt"
    (source / "best.pt").unlink()
    with pytest.raises(FileNotFoundError, match="best.pt"):
        resolve_checkpoint_path(str(source), operation="transfer")


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


@pytest.mark.parametrize("asset", ["yaml", "image", "centerline", "walls"])
def test_application_recovery_rejects_changed_map_yaml_hashes(asset):
    from core.provenance import collect_map_protocols
    from core.scenario import load_and_expand_scenario

    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    metadata = {"configuration": scenario, "provenance": {
        "map_protocols": collect_map_protocols(scenario["environment"])}}
    validate_resume_experiment(metadata, scenario)
    first = next(iter(metadata["provenance"]["map_protocols"].values()))
    if asset == "yaml":
        first["yaml_sha256"] = "changed_map_contents"
    else:
        first["assets"][asset]["sha256"] = "changed_map_contents"
    with pytest.raises(ValueError, match="map_protocols"):
        validate_resume_experiment(metadata, scenario)


def test_old_provenance_recovers_with_its_existing_yaml_checks():
    from core.provenance import collect_map_protocols
    from core.scenario import load_and_expand_scenario
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    maps = collect_map_protocols(scenario["environment"])
    for details in maps.values():
        details.pop("assets")
    validate_resume_experiment({"configuration": scenario, "provenance": {
        "version": "1.0", "map_protocols": maps}}, scenario)


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


@pytest.mark.parametrize("resumed_limit, expected_steps", [(None, 140), (128, 128), (120, 120)])
def test_diagnostic_stop_completes_updates_evaluates_and_resumes_full_budget(
        tmp_path, resumed_limit, expected_steps):
    # Scale samples per update to four; retain 30 updates and evaluation at 10/20/30.
    capture = Capture(False)
    original, fixed = trainer(capture, n_epochs=1, batch_size=4,
                              learning_rate=.001, lr_schedule="linear", learning_rate_end=.0001)
    original = OnPolicyTrainer(original.task, original.agent, hooks=[capture],
                               scenario={"experiment": {"stop_after_steps": 120}})
    evaluations = []

    class Evaluator:
        def evaluate(self):
            evaluations.append((original._updates, original.collected_steps))
            return {"completion_rate": 0., "collision_rate": 1., "mean_net_progress": 0.}

    original.hooks.extend([
        CheckpointHook(original.agent, tmp_path, save_every_steps=10),
        EvaluationCheckpointHook(original.agent, tmp_path, Evaluator(), evaluate_every=100,
                                 evaluate_every_steps=40),
    ])
    original.train(total_steps=140)
    assert original.collected_steps == fixed.calls == 120
    assert original._updates == original._optimizer_steps == 30
    assert original._pending_learning_steps == 0 and not original.budget_reached()
    assert evaluations == [(10, 40), (20, 80), (30, 120)]
    saved = read_checkpoint(tmp_path / "latest.pt")
    assert not (tmp_path / "final.pt").exists()
    state = saved["training"]
    assert state["budget"] == {"total_steps": 140, "n_episodes": 0}
    assert state["schedule_progress"] == pytest.approx(120 / 140)
    assert state["progress"]["environment_steps"] == 120
    assert state["progress"]["updates"] == state["progress"]["optimizer_steps"] == 30
    assert state["hooks"]["EvaluationCheckpointHook"]["evaluation_count"] == 3
    expected_lr = .001 + (120 / 140) * (.0001 - .001)
    assert original.agent.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)

    resumed_capture = Capture(False)
    recovered, resumed_fixed = trainer(resumed_capture, n_epochs=1, batch_size=4,
                                      learning_rate=.001, lr_schedule="linear", learning_rate_end=.0001)
    recovered = OnPolicyTrainer(recovered.task, recovered.agent, hooks=[resumed_capture],
                                scenario={"experiment": {"stop_after_steps": resumed_limit}})
    recovered_evaluations = []

    class ResumedEvaluator:
        def evaluate(self):
            recovered_evaluations.append(recovered.collected_steps)
            return {"completion_rate": 0., "collision_rate": 1., "mean_net_progress": 0.}

    recovered.hooks.extend([
        CheckpointHook(recovered.agent, tmp_path / "resume", save_every_steps=10),
        EvaluationCheckpointHook(recovered.agent, tmp_path / "resume", ResumedEvaluator(),
                                 evaluate_every=100, evaluate_every_steps=40),
    ])
    recovery_state, _ = recovered.agent.resume_training(tmp_path / "latest.pt")
    recovered.restore_training_state(recovery_state)
    for name in ("actor", "critic"):
        for key, value in getattr(original.agent, name).state_dict().items():
            torch.testing.assert_close(value, getattr(recovered.agent, name).state_dict()[key], rtol=0, atol=0)
    before, after = original.agent.optimizer.state_dict(), recovered.agent.optimizer.state_dict()
    assert before["param_groups"] == after["param_groups"]
    for key, values in before["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, after["state"][key][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="total_steps"):
        recovered.train(total_steps=141)
    recovered.train(total_steps=140)
    assert recovered.collected_steps == expected_steps
    assert resumed_fixed.calls == expected_steps - 120
    assert recovered._updates == recovered._optimizer_steps == expected_steps // 4
    assert [row["train/environment_steps"] for row in resumed_capture.updates] == list(
        range(124, expected_steps + 1, 4))
    assert recovered_evaluations == []  # Evaluation at update 30 is not repeated.
    assert recovered.agent.optimizer.param_groups[0]["lr"] == pytest.approx(
        .001 + (expected_steps / 140) * (.0001 - .001))


def test_diagnostic_limit_between_updates_waits_for_complete_update():
    capture = Capture(False)
    training, fixed = trainer(capture, n_epochs=1, batch_size=4)
    training.stop_after_steps = 5
    training.train(total_steps=20)
    assert training.collected_steps == fixed.calls == 8
    assert training._updates == 2 and training.checkpoint_ready


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
    assert continued.collected_steps == 5 and len(capture.steps) == 8
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


def modular_transfer_pair(preset="cnn1d_base", head="mlp", actor_mode="shared", beams=108, *, native_schedule=False):
    """Small models with real resolved contracts and deliberately different critics."""
    from agents.common.mappo_policy import MAPPOPolicy
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from core.scenario import load_and_expand_scenario
    from test_torchrl_mappo import BOUNDS, IDS, params as team_params
    from wrappers.observations.composer import ObservationComposer

    configs, composers = [], []
    for path in ("ppo_lap_completion_pretrain", "mappo_2v2_completion_scratch"):
        choices = [f"network={preset}", f"environment.lidar_beams={beams}",
                   "network.actor_hidden_dims=[8,6]", "network.critic_hidden_dims=[7]"]
        if "cnn" in preset:
            choices += ["network.actor.encoder.channels=[4,8]", "network.actor.encoder.pool_bins=3",
                        "network.actor.encoder.lidar_features=8", "network.actor.encoder.vehicle_features=4",
                        "network.actor.encoder.neighbor_features=4", "network.actor.encoder.vehicle_hidden_dims=[5]",
                        "network.actor.encoder.neighbor_hidden_dims=[5]"]
            if preset == "rescnn1d_base":
                choices += ["network.actor.encoder.num_blocks=1"]
        if head == "resmlp":
            choices += ["network.actor.architecture=resmlp", "++network.actor.width=16", "++network.actor.num_blocks=2"]
        config = load_and_expand_scenario(f"scenarios/{path}.yaml", overrides=choices)
        configs.append(config)
        composers.append(ObservationComposer.from_config(config["agents"]["car_0"]["observation"], config["environment"]))
    source_config, target_config = configs
    # A CNN source critic must never restrict the independently chosen global-state critic.
    if "cnn" in preset:
        source_config["network"]["critic"]["encoder"] = deepcopy(source_config["network"]["actor"]["encoder"])
        target_config["network"]["actor"]["encoder"].update(neighbor_features=6, neighbor_hidden_dims=[7])
    target_config["network"]["critic"].update(architecture="resmlp", width=13, num_blocks=1)
    common = dict(device="cpu", learning_rate=.001, lr_schedule="linear", learning_rate_end=.0001)
    source_params = dict(common, network=source_config["network"], _observation_contract=composers[0].contract)
    target_params = team_params(**common, network=target_config["network"], actor_mode=actor_mode,
                               _observation_contract=composers[1].contract)
    if native_schedule:
        target_params.update(lr_schedule="step", learning_rate_end=None,
                             lr_scheduler_kwargs={"step_size": 1, "gamma": .5})
    torch.manual_seed(101)
    source = PPOPolicy(composers[0].obs_dim, *BOUNDS, source_params, training=True)
    with torch.no_grad():
        source.actor.log_std.copy_(torch.tensor([-.4, .3]))
    source.set_training_progress(.8)
    torch.manual_seed(202)
    target = TorchRLMAPPOAgent(composers[1].obs_dim, 3, *BOUNDS, IDS, target_params)
    torch.manual_seed(202)
    control = MAPPOPolicy(composers[1].obs_dim, 3, *BOUNDS, IDS, target_params)
    for key, value in control.critic.state_dict().items():
        torch.testing.assert_close(value, target.critic.state_dict()[key], rtol=0, atol=0)
    return source, target


TRANSFER_ARCHITECTURES = [("mlp_base", "mlp"), ("mlp_base", "resmlp"),
                          ("cnn1d_base", "mlp"), ("rescnn1d_base", "mlp"),
                          ("cnn1d_base", "resmlp"), ("rescnn1d_base", "resmlp")]


@pytest.mark.parametrize("preset,head", TRANSFER_ARCHITECTURES)
@pytest.mark.parametrize("actor_mode", ["shared", "independent"])
@pytest.mark.parametrize("beams", [108, 1080])
def test_modular_transfer_preserves_distribution_weights_and_fresh_critic(preset, head, actor_mode, beams, tmp_path):
    source, target = modular_transfer_pair(preset, head, actor_mode, beams)
    path = tmp_path / "ppo.pt"
    source.save_checkpoint(path, provenance={"run_id": "source-ppo", "seed": 101})
    source_bytes = path.read_bytes()
    before_actor = deepcopy(target.actor.state_dict())
    before_critic = deepcopy(target.critic.state_dict())
    optimizer = deepcopy(target.optimizer.state_dict())
    rng = torch.random.get_rng_state().clone()
    target.initialize_from_checkpoint(path, scope="actor_only", observation_extension="frenet_neighbors")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert target.optimizer.state_dict() == optimizer and not target.optimizer.state
    assert target.optimizer.param_groups[0]["lr"] == target.lr != source.optimizer.param_groups[0]["lr"]
    assert all(parameter.requires_grad for parameter in target.actor.parameters())
    assert all(parameter.requires_grad for parameter in target.critic.parameters())
    actors = list(target.actor.actors.values()) if actor_mode == "independent" else [target.actor]
    key = "net.0.weight" if head == "mlp" else "net.input_projection.0.weight"
    old_width = source.actor.encoder.output_dim
    source_state = source.actor.state_dict()
    for i, actor in enumerate(actors):
        prefix = f"actors.{target.agent_ids[i]}." if actor_mode == "independent" else ""
        for name, weight in actor.state_dict().items():
            if name.startswith("encoder.neighbors."):
                torch.testing.assert_close(weight, before_actor[prefix + name], rtol=0, atol=0)
            elif name == key:
                torch.testing.assert_close(weight[:, :old_width], source_state[name], rtol=0, atol=0)
                assert torch.count_nonzero(weight[:, old_width:]) == 0
            else:
                torch.testing.assert_close(weight, source_state[name], rtol=0, atol=0)
        observations = torch.randn(7, source.obs_dim)
        mean, std = source.actor(observations)
        for scale in (0., 1., 100.):
            neighbors = torch.randn(7, target.obs_dim - source.obs_dim) * scale
            full = torch.cat((observations, neighbors), dim=-1)
            transferred_mean, transferred_std = actor(full)
            torch.testing.assert_close(transferred_mean, mean, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(transferred_std, std, rtol=0, atol=0)
            torch.testing.assert_close(actor.get_action(full, deterministic=True)[0], mean.tanh(), rtol=1e-6, atol=1e-6)
    if actor_mode == "independent":
        assert next(actors[0].parameters()).data_ptr() != next(actors[1].parameters()).data_ptr()
        data = torch.randn(2, target.obs_dim)
        routed_mean, _ = target.actor(data, adapter_indices=torch.tensor([0, 1]))
        torch.testing.assert_close(routed_mean, source.actor(data[:, :source.obs_dim])[0], rtol=1e-6, atol=1e-6)
    for name, weight in target.critic.state_dict().items():
        torch.testing.assert_close(weight, before_critic[name], rtol=0, atol=0)
    saved = tmp_path / "mappo.pt"
    target.save_checkpoint(saved)
    payload = read_checkpoint(saved)
    assert payload["schema_version"] == 1
    assert payload["metadata"]["structure"]["actor_mode"] == actor_mode
    assert payload["metadata"]["adaptation"]["source_checkpoint"]["provenance"]["run_id"] == "source-ppo"
    target.load_for_evaluation(saved)
    assert target.source_checkpoint["sha256"] == read_checkpoint(path)["_identity"]["sha256"]
    assert path.read_bytes() == source_bytes


@pytest.mark.parametrize("preset,head", TRANSFER_ARCHITECTURES[1:])
def test_transferred_actor_and_new_neighbor_connections_learn_with_native_mappo(preset, head, tmp_path):
    from test_torchrl_mappo import IDS, transition

    source, target = modular_transfer_pair(preset, head)
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    target.initialize_from_checkpoint(path, scope="actor_only", observation_extension="frenet_neighbors")
    projection = target.actor.net[0] if head == "mlp" else target.actor.net.input_projection[0]
    width = source.actor.encoder.output_dim
    for step in range(2):
        target.optimizer.zero_grad(set_to_none=True)
        x = torch.randn(9, target.obs_dim)
        loss = (target.actor(x)[0] - torch.randn(9, 2)).square().mean()
        loss += (target.critic(torch.randn(9, 3)) - 1).square().mean()
        loss.backward()
        assert torch.count_nonzero(projection.weight.grad[:, :width]) > 0
        assert torch.count_nonzero(projection.weight.grad[:, width:]) > 0
        if "cnn" in preset:
            assert any(torch.count_nonzero(p.grad) > 0 for p in target.actor.encoder.lidar_backbone.parameters())
            neighbor_gradient = sum(p.grad.abs().sum() for p in target.actor.encoder.neighbors.parameters())
            assert (neighbor_gradient == 0) if step == 0 else (neighbor_gradient > 0)
        assert any(torch.count_nonzero(p.grad) > 0 for p in target.critic.parameters())
        target.optimizer.step()
    observations = {aid: np.random.default_rng(i).normal(size=target.obs_dim).astype(np.float32) for i, aid in enumerate(IDS)}
    data = torch.stack([transition(target, IDS, step, 1., terminal=step == 2, observations=observations) for step in range(3)])
    metrics = target.update(data)
    assert metrics["train/optimizer_steps"] > 0
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("field", ["beams", "normalization", "range", "preview", "kernel", "stride", "channels", "blocks",
    "head_width", "head_activation", "features", "physics", "actions", "distribution", "missing_contract", "malformed_contract",
    "extension", "tensor", "dtype", "extra_key", "missing_key", "segments", "source_critic", "late_actor"])
def test_modular_transfer_rejections_leave_all_destination_weights_unchanged(field, tmp_path):
    source, target = modular_transfer_pair("rescnn1d_base", "resmlp", "independent")
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    payload = read_checkpoint(path)
    meta = payload["metadata"]
    encoder = meta["network"]["actor"]["encoder"]
    contract = meta["contracts"]["observations"]["policy"]["contract"]
    extension = "frenet_neighbors"
    if field == "beams":
        contract["lidar_beams"] += 1
    elif field == "normalization":
        contract["observation"]["lidar"]["normalize"] = False
    elif field == "range":
        contract["lidar_range"] += 1
    elif field == "preview":
        contract["observation"]["frenet_vehicle_track"]["points"] += 1
    elif field in {"kernel", "stride", "channels", "blocks", "features"}:
        option = {"kernel": "kernel_size", "stride": "stride", "channels": "channels", "blocks": "num_blocks", "features": "lidar_features"}[field]
        encoder[option] = [5, 8] if field == "channels" else encoder[option] + (2 if field == "kernel" else 1)
    elif field == "head_width":
        meta["network"]["actor"]["width"] += 1
    elif field == "head_activation":
        meta["network"]["actor"]["activation"] = "relu"
    elif field == "physics":
        meta["contracts"]["physics"] = {"changed": True}
    elif field == "actions":
        meta["contracts"]["actions"]["low"][0] -= 1
    elif field == "distribution":
        meta["structure"]["distribution"] = "normal"
    elif field == "missing_contract":
        meta["contracts"]["observations"]["policy"]["contract"] = None
    elif field == "malformed_contract":
        meta["contracts"]["observations"]["policy"]["contract"] = {"version": 1}
    elif field == "extension":
        extension = "arbitrary_features"
    elif field in {"tensor", "dtype"}:
        key = "net.input_projection.0.weight"
        payload["models"]["actor"][key] = torch.zeros(1) if field == "tensor" else payload["models"]["actor"][key].double()
    elif field == "extra_key":
        payload["models"]["actor"]["encoder.neighbors.unexpected"] = torch.zeros(1)
    elif field == "missing_key":
        payload["models"]["actor"].pop("log_std")
    elif field == "segments":
        meta["structure"]["observation_segments"]["actor"]["policy"]["lidar"][1] += 1
    elif field == "source_critic":
        payload["models"]["critic"] = {}
    else:
        # A mismatch discovered after preparing actor zero must still be atomic.
        target.actor.actors[target.agent_ids[1]].net.output = torch.nn.Linear(16, 3)
    atomic_save(payload, path)
    before = {role: deepcopy(getattr(target, role).state_dict()) for role in ("actor", "critic")}
    optimizer = deepcopy(target.optimizer.state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        target.initialize_from_checkpoint(path, scope="actor_only", observation_extension=extension)
    for role in before:
        for name, weight in before[role].items():
            torch.testing.assert_close(weight, getattr(target, role).state_dict()[name], rtol=0, atol=0)
    assert target.optimizer.state_dict() == optimizer
    assert target.source_checkpoint is None
    assert torch.equal(rng, torch.random.get_rng_state())



def test_transfer_keeps_a_fresh_native_scheduler_and_optimizer(tmp_path):
    source, target = modular_transfer_pair(native_schedule=True)
    source.optimizer.zero_grad()
    sum(p.square().mean() for p in source.actor.parameters()).backward()
    source.optimizer.step()
    assert source.optimizer.state
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    schedule = deepcopy(target.scheduler.state_dict())
    target.initialize_from_checkpoint(path, scope="actor_only", observation_extension="frenet_neighbors")
    assert not target.optimizer.state
    assert target.scheduler.state_dict() == schedule
    assert target.optimizer.param_groups[0]["lr"] == target.lr


def test_transfer_rejects_consistent_but_different_lidar_resolutions(tmp_path):
    source, _ = modular_transfer_pair(beams=108)
    _, target = modular_transfer_pair(beams=1080)
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    before = {role: deepcopy(getattr(target, role).state_dict()) for role in ("actor", "critic")}
    with pytest.raises(ValueError, match="observation|Frenet"):
        target.initialize_from_checkpoint(path, scope="actor_only", observation_extension="frenet_neighbors")
    for role in before:
        for name, weight in before[role].items():
            torch.testing.assert_close(weight, getattr(target, role).state_dict()[name], rtol=0, atol=0)


def test_evaluation_only_transfer_with_a_linear_schedule_needs_no_optimizer(tmp_path):
    source = agent(hidden_dims=[4])
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    destination = PPOPolicy(1, source.action_low, source.action_high, {"hidden_dims": [4],
        "device": "cpu", "lr_schedule": "linear", "learning_rate_end": .0001})
    destination.initialize_from_checkpoint(path, scope="actor_and_critic")
    assert destination.optimizer is None
    for role in ("actor", "critic"):
        for key, value in getattr(source, role).state_dict().items():
            torch.testing.assert_close(value, getattr(destination, role).state_dict()[key], rtol=0, atol=0)


def modular_lora_pair(preset="cnn1d_base", head="mlp", beams=108, *, mode="per_agent", train_log_std=True):
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from test_torchrl_mappo import BOUNDS, IDS, params

    source, control = modular_transfer_pair(preset, head, beams=beams)
    options = params(network=control.network_config, _observation_contract=control.observation_contract,
        actor_mode="shared", learning_rate=.02, optimizer="adamw", optimizer_kwargs={"weight_decay": .2},
        lora={"target": "actor_head", "mode": mode, "rank": 2, "alpha": 3.,
              "train_log_std": train_log_std, "per_agent_log_std": mode == "per_agent"})
    torch.manual_seed(202)
    target = TorchRLMAPPOAgent(control.obs_dim, control.global_state_dim, *BOUNDS, IDS, options)
    for name, value in control.critic.state_dict().items():
        torch.testing.assert_close(value, target.critic.state_dict()[name], rtol=0, atol=0)
    return source, target, options


@pytest.mark.parametrize("preset,head", TRANSFER_ARCHITECTURES)
@pytest.mark.parametrize("beams", [108, 1080])
@pytest.mark.parametrize("mode,train_log_std", [("per_agent", True), ("shared", False)])
def test_modular_lora_initial_equivalence_freezing_and_accounting(preset, head, beams, mode, train_log_std, tmp_path):
    from agents.common.lora import parameter_accounting

    source, target, _ = modular_lora_pair(preset, head, beams, mode=mode, train_log_std=train_log_std)
    source.save_checkpoint(tmp_path / "source.pt", provenance={"run_id": "ppo-lora-source"})
    neighbor_before = deepcopy(target.actor.encoder.neighbors.state_dict()) if "cnn" in preset else {}
    target.initialize_from_checkpoint(tmp_path / "source.pt", scope="actor_only", observation_extension="frenet_neighbors")
    base = target.actor.base_state_dict()
    first = "net.0.weight" if head == "mlp" else "net.input_projection.0.weight"
    width = source.actor.encoder.output_dim
    for name, weight in source.actor.state_dict().items():
        torch.testing.assert_close(base[name][:, :width] if name == first else base[name], weight, rtol=0, atol=0)
    assert torch.count_nonzero(base[first][:, width:]) == 0
    first_key = next(iter(target.actor.adapters[0]))
    assert target.actor.adapters[0][first_key].A.shape[1] == target.actor.encoder.output_dim
    assert all(torch.count_nonzero(adapter.B) == 0 for bank in target.actor.adapters for adapter in bank.values())
    if "cnn" in preset:
        for name, value in neighbor_before.items():
            torch.testing.assert_close(value, target.actor.encoder.neighbors.state_dict()[name], rtol=0, atol=0)
        assert all(not p.requires_grad for p in target.actor.encoder.lidar_backbone.parameters())
        assert all(not p.requires_grad for p in target.actor.encoder.lidar_projection.parameters())
        assert all(not p.requires_grad for p in target.actor.encoder.vehicle_track.parameters())
        assert all(p.requires_grad for p in target.actor.encoder.neighbors.parameters())
    assert all(not p.requires_grad for p in target.actor.net.parameters())
    assert all(p.requires_grad for p in target.actor.adapters.parameters())
    assert all(p.requires_grad for p in target.critic.parameters())
    for _ in range(3):
        driving = torch.randn(6, source.obs_dim)
        expected_mean, expected_std = source.actor(driving)
        for scale in (0., 1., 30.):
            obs = torch.cat((driving, torch.randn(6, target.obs_dim - source.obs_dim) * scale), dim=-1)
            kwargs = {"adapter_indices": torch.tensor([0, 1, 1, 0, 1, 0])} if mode == "per_agent" else {}
            mean, std = target.actor(obs, **kwargs)
            torch.testing.assert_close(mean, expected_mean, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(std.expand_as(mean), expected_std.expand_as(mean), rtol=0, atol=0)
            actions = target.actor.get_action(obs, deterministic=True, **kwargs)[0]
            torch.testing.assert_close(actions, expected_mean.tanh(), rtol=1e-6, atol=1e-6)
    counts = parameter_accounting(target)
    assert counts["actor_total"] == counts["actor_trainable"] + counts["actor_frozen"]
    assert counts["actor_trainable"] == counts["lora_trainable"] + counts["neighbor_encoder_trainable"] + counts["exploration_trainable"]
    assert counts["exploration_trainable"] == (4 if train_log_std else 0)
    assert counts["critic_trainable"] == sum(p.numel() for p in target.critic.parameters())
    frozen_ids = {id(p) for p in target.actor.parameters() if not p.requires_grad}
    optimizer_ids = {id(p) for group in target.optimizer.param_groups for p in group["params"]}
    assert not frozen_ids.intersection(optimizer_ids)
    assert optimizer_ids == {id(p) for module in (target.actor, target.critic) for p in module.parameters() if p.requires_grad}
    target.save_checkpoint(tmp_path / "lora.pt")
    metadata = read_checkpoint(tmp_path / "lora.pt")["metadata"]
    assert metadata["parameter_counts"] == counts
    assert target.source_checkpoint["network"] == source.network_config
    if not target.actor.legacy:
        assert metadata["adaptation"]["lora"]["target_modules"] == target.actor.target_modules
        assert metadata["adaptation"]["lora"]["freezing"]["neighbor_encoder_shared"]


@pytest.mark.parametrize("preset,head", TRANSFER_ARCHITECTURES)
def test_modular_lora_learns_neighbors_without_changing_frozen_weights(preset, head, tmp_path):
    from test_torchrl_mappo import IDS, transition

    source, target, _ = modular_lora_pair(preset, head)
    source_path = tmp_path / "source.pt"
    source.save_checkpoint(source_path)
    source_bytes = source_path.read_bytes()
    target.initialize_from_checkpoint(source_path, scope="actor_only", observation_extension="frenet_neighbors")
    frozen = {name: p.detach().clone() for name, p in target.actor.named_parameters() if not p.requires_grad}
    before_banks = [deepcopy(bank.state_dict()) for bank in target.actor.adapters]
    neighbor_before = deepcopy(target.actor.encoder.neighbors.state_dict()) if "cnn" in preset else {}
    critic_before = deepcopy(target.critic.state_dict())
    driving = torch.randn(12, source.obs_dim)
    extra = torch.randn(12, target.obs_dim - source.obs_dim)
    observations = torch.cat((driving, extra), dim=-1)
    teacher = torch.stack((extra[:, 0].tanh(), extra[:, 1].tanh()), dim=-1)
    saw_neighbor_gradient = False
    for step in range(16):
        target.optimizer.zero_grad(set_to_none=True)
        bank_id = 0 if step == 0 else step % 2
        indices = torch.full((12,), bank_id, dtype=torch.long)
        mean, _ = target.actor(observations, adapter_indices=indices)
        loss = (mean - teacher).square().mean() + (target.critic(torch.randn(12, 3)) - .7).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in target._optim_parameters if p.grad is not None)
        if "cnn" in preset:
            magnitude = sum(p.grad.abs().sum() for p in target.actor.encoder.neighbors.parameters())
            if step == 0:
                assert magnitude == 0
            saw_neighbor_gradient |= bool(magnitude > 0)
        target.optimizer.step()
        if step == 0:
            for key, value in before_banks[1].items():
                torch.testing.assert_close(value, target.actor.adapters[1].state_dict()[key], rtol=0, atol=0)
    for i, bank in enumerate(target.actor.adapters):
        assert any(not torch.equal(value, before_banks[i][key]) for key, value in bank.state_dict().items() if key.endswith(".A"))
        assert any(not torch.equal(value, before_banks[i][key]) for key, value in bank.state_dict().items() if key.endswith(".B"))
    if "cnn" in preset:
        assert saw_neighbor_gradient
        assert any(not torch.equal(value, neighbor_before[key]) for key, value in target.actor.encoder.neighbors.state_dict().items())
    with torch.no_grad():
        a = target.actor(observations, adapter_indices=torch.zeros(12, dtype=torch.long))[0]
        b = target.actor(torch.cat((driving, -extra), dim=-1), adapter_indices=torch.zeros(12, dtype=torch.long))[0]
    assert (a - b).abs().max() > 1e-5
    assert any(not torch.equal(value, critic_before[key]) for key, value in target.critic.state_dict().items())
    # Native MAPPO scoring and optimization must use the same encoded adapter path.
    obs = {aid: observations[i].numpy() for i, aid in enumerate(IDS)}
    data = torch.stack([transition(target, IDS, i, i + 1, terminal=i == 2, observations=obs) for i in range(3)])
    assert target.update(data)["train/optimizer_steps"] > 0
    for name, value in frozen.items():
        torch.testing.assert_close(value, dict(target.actor.named_parameters())[name], rtol=0, atol=0)
    assert source_path.read_bytes() == source_bytes


@pytest.mark.parametrize("preset,head", TRANSFER_ARCHITECTURES)
def test_modular_lora_checkpoint_routing_pickle_and_historical_layout(preset, head, tmp_path):
    import pickle
    from agents.common.checkpoints import evaluation_routing
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from test_torchrl_mappo import BOUNDS, IDS

    source, target, options = modular_lora_pair(preset, head)
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    target.initialize_from_checkpoint(path, scope="actor_only", observation_extension="frenet_neighbors")
    with torch.no_grad():
        for bank in target.actor.adapters:
            for residual in bank.values():
                residual.B.normal_(std=.03)
    saved = tmp_path / "lora.pt"
    target.save_checkpoint(saved)
    payload = read_checkpoint(saved)
    if target.actor.legacy:
        assert payload["metadata"]["adaptation"]["lora"]["version"] == 1
        assert "adapters.0.0.A" in payload["models"]["actor"]
        assert not any(key.startswith("encoder.") for key in payload["models"]["actor"])
        # Original schema-1 files did not contain these additive provenance/count fields.
        payload["metadata"].pop("parameter_counts")
        payload["metadata"]["adaptation"]["source_checkpoint"].pop("network")
        atomic_save(payload, saved)
    config = {**options, **evaluation_routing(payload)}
    loaded = TorchRLMAPPOAgent(target.obs_dim, 3, *BOUNDS, IDS, config)
    loaded.load_for_evaluation(saved)
    x = torch.randn(5, target.obs_dim)
    indices = torch.tensor([0, 1, 1, 0, 1])
    expected = target.actor(x, adapter_indices=indices)
    for actor in (loaded.actor, pickle.loads(pickle.dumps(loaded.actor))):
        for actual, original in zip(actor(x, adapter_indices=indices), expected):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert loaded.source_checkpoint["sha256"] == target.source_checkpoint["sha256"]


@pytest.mark.parametrize("field", ["rank", "target", "freeze", "layout", "source", "tensor", "targets"])
def test_modular_lora_checkpoint_rejection_is_atomic(field, tmp_path):
    source, target, _ = modular_lora_pair("rescnn1d_base", "resmlp")
    source.save_checkpoint(tmp_path / "source.pt")
    target.initialize_from_checkpoint(tmp_path / "source.pt", scope="actor_only", observation_extension="frenet_neighbors")
    target.save_checkpoint(tmp_path / "lora.pt")
    payload = read_checkpoint(tmp_path / "lora.pt")
    contract = payload["metadata"]["adaptation"]["lora"]
    if field == "rank":
        contract["rank"] += 1
    elif field == "target":
        contract["target"] = "encoder"
    elif field == "freeze":
        contract["freezing"]["base_head"] = False
    elif field == "layout":
        payload["metadata"]["structure"]["observation_segments"]["actor"]["car_0"]["lidar"][1] += 1
    elif field == "source":
        payload["metadata"]["adaptation"]["source_checkpoint"]["sha256"] = "0" * 64
    elif field == "tensor":
        name = next(key for key in payload["models"]["actor"] if key.startswith("adapters."))
        payload["models"]["actor"][name] = torch.zeros(1)
    else:
        contract["target_modules"] = ["net.output"]
    atomic_save(payload, tmp_path / "lora.pt")
    before = {role: deepcopy(getattr(target, role).state_dict()) for role in ("actor", "critic")}
    for operation in ("load_for_evaluation", "resume_training"):
        if operation == "resume_training":
            # Model-only resume rejects before mutation as usual.
            pattern = "model-only"
        else:
            pattern = "Incompatible"
        with pytest.raises(ValueError, match=pattern):
            getattr(target, operation)(tmp_path / "lora.pt")
        for role in before:
            for key, value in before[role].items():
                torch.testing.assert_close(value, getattr(target, role).state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("preset", ["cnn1d_base", "rescnn1d_base"])
def test_lora_rejects_pretrained_neighbor_branches_without_mutation(preset, tmp_path):
    from test_torchrl_mappo import BOUNDS

    _, target, _ = modular_lora_pair(preset)
    source = PPOPolicy(target.obs_dim, *BOUNDS, {"network": target.network_config,
        "device": "cpu", "_observation_contract": target.observation_contract})
    source.save_checkpoint(tmp_path / "source.pt")
    before = deepcopy(target.actor.state_dict())
    with pytest.raises(ValueError, match="driving-only PPO source"):
        target.initialize_from_checkpoint(tmp_path / "source.pt", scope="actor_only")
    for key, value in before.items():
        torch.testing.assert_close(value, target.actor.state_dict()[key], rtol=0, atol=0)


def test_modular_lora_initialization_is_deterministic_across_trials(tmp_path):
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from test_torchrl_mappo import BOUNDS, IDS

    source, original, options = modular_lora_pair("rescnn1d_base", "resmlp")
    source.save_checkpoint(tmp_path / "source.pt")
    states = []
    for _ in range(2):
        torch.manual_seed(81)
        target = TorchRLMAPPOAgent(original.obs_dim, 3, *BOUNDS, IDS, options)
        target.initialize_from_checkpoint(tmp_path / "source.pt", scope="actor_only", observation_extension="frenet_neighbors")
        states.append(deepcopy(target.actor.state_dict()))
    for key, value in states[0].items():
        torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)
