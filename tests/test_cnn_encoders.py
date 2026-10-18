"""CNN layout, fusion, native optimization/recovery and compatibility checks."""
from copy import deepcopy
import math
import multiprocessing

import numpy as np
import pytest
import torch
from torch import nn

from agents.common.checkpoints import read_checkpoint, restore_rng
from agents.common.encoders import build_encoder, resolve_encoder_config, ResidualConv1DBlock
from agents.common.lora import LoRAActor
from agents.common.networks import build_actor, build_critic, model_spec, resolve_network_config
from agents.common.observations import observation_segments
from application.configuration import validate_experiment_scenario
from core.scenario import load_and_expand_scenario, ScenarioError
from core.task_builder import create_race_task
from training.algorithms import create_learner, learner_params
from training.hooks import CheckpointHook
from training.on_policy import OnPolicyTrainer
from utils.torch_io import atomic_save
from wrappers.observations.composer import ObservationComposer
from test_torchrl_ppo import Capture


PRESETS = ["cnn1d_base", "rescnn1d_base"]


def scenario(algorithm="ppo", preset="cnn1d_base", *, small=False, overrides=()):
    choices = [f"network={preset}", "environment.max_steps=3", "execution.params.device=cpu"]
    if small:
        choices += ["network.actor.encoder.channels=[4,8]", "network.actor.encoder.lidar_features=8",
                    "network.actor.encoder.vehicle_features=4", "network.actor.encoder.neighbor_features=4",
                    "network.actor.encoder.vehicle_hidden_dims=[4]", "network.actor.encoder.neighbor_hidden_dims=[4]",
                    "network.actor.encoder.pool_bins=4", "network.actor_hidden_dims=[8]", "network.critic_hidden_dims=[12]",
                    f"algorithm.params.n_steps={4 if algorithm == 'ppo' else 8}",
                    "++algorithm.params.frames_per_batch=4", "algorithm.params.n_epochs=1",
                    "algorithm.params.batch_size=4"]
        if preset == "rescnn1d_base":
            choices += ["network.actor.encoder.num_blocks=1"]
    return load_and_expand_scenario(f"scenarios/{algorithm}_serial_profile.yaml", overrides=[*choices, *overrides])


def composer(config):
    agent = next(agent for agent in config["agents"].values() if agent["trainable"])
    return ObservationComposer.from_config(agent["observation"], config["environment"])


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
@pytest.mark.parametrize("beams", [108, 1080])
def test_presets_shapes_seeded_initialization_and_backward(preset, algorithm, beams):
    config = scenario(algorithm, preset, overrides=[f"environment.lidar_beams={beams}"])
    validate_experiment_scenario(config)
    task = create_race_task(config)
    try:
        params = learner_params(config, task.spec, algorithm)
        torch.manual_seed(17)
        policy = create_learner(algorithm, task.spec, params, training=False)
        torch.manual_seed(17)
        repeated = create_learner(algorithm, task.spec, params, training=False)
        for role in ("actor", "critic"):
            for key, weight in getattr(policy, role).state_dict().items():
                torch.testing.assert_close(weight, getattr(repeated, role).state_dict()[key], rtol=0, atol=0)
        actor = policy.actor.actors[task.spec.agent_ids[0]] if algorithm == "mappo" else policy.actor
        dim = task.spec.observation_dims[task.spec.agent_ids[0]]
        x = torch.randn(2, 3, dim, requires_grad=True)
        mean, std = actor(x)
        assert mean.shape == (2, 3, 2) and std.shape == (2,)
        assert actor.encoder.output_dim == (192 if algorithm == "ppo" else 224)
        assert actor.net[0].in_features == actor.encoder.output_dim
        assert actor.encoder.segments["lidar"] == [0, beams]
        critic_dim = dim if algorithm == "ppo" else policy.critic_input_dim
        value = policy.critic(torch.randn(2, 3, critic_dim))
        assert value.shape == (2, 3)
        (mean.square().mean() + std.mean() + value.square().mean()).backward()
        assert torch.isfinite(x.grad).all()
        for model in (actor.encoder, actor.net, policy.critic):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert resolve_network_config({"network": policy.network_config}, default_hidden_dims=[]) == policy.network_config
    finally:
        task.close()


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("neighbors", [False, True])
def test_configurable_segments_branch_isolation_and_flat_values(preset, neighbors):
    config = scenario("mappo", preset, small=True, overrides=[
        "environment.lidar_beams=73", "observation.observation.frenet_vehicle_track.points=7",
        f"observation.observation.frenet_neighbors.enabled={str(neighbors).lower()}"])
    comp = composer(config)
    encoder = build_encoder(comp.obs_dim, config["network"]["actor"]["encoder"], observation_contract=comp.contract)
    raw = {"lidar": np.linspace(0, 20, 73, dtype=np.float32),
           "wheel_speed_reference_rate": 0., "wheel_speed_reference": 0., "wheel_speed": 0.}
    wrapped = comp.wrap(raw, {"agent_id": "car_0"})
    np.testing.assert_array_equal(wrapped[:73], np.minimum(raw["lidar"] * np.float32(.1), 1))
    assert encoder.segments["vehicle_track"] == [73, 97]
    assert (encoder.neighbors is not None) == neighbors
    values = torch.arange(2 * comp.obs_dim, dtype=torch.float32).reshape(2, comp.obs_dim)
    original = values.clone()
    received = {}
    handles = []
    for name, module in (("lidar", encoder.lidar_backbone), ("vehicle_track", encoder.vehicle_track),
                         ("neighbors", encoder.neighbors)):
        if module is not None:
            def capture(module, args, name=name):
                received[name] = args[0].detach().clone()
            handles.append(module.register_forward_pre_hook(capture))
    try:
        assert encoder(values).shape == (2, 16 if neighbors else 12)
        for name, tensor in received.items():
            start, stop = encoder.segments[name]
            torch.testing.assert_close(tensor.squeeze(1) if name == "lidar" else tensor,
                                       values[:, start:stop], rtol=0, atol=0)
        torch.testing.assert_close(values, original, rtol=0, atol=0)
        assert encoder(values[0]).shape == (encoder.output_dim,)
        with pytest.raises(ValueError, match="Expected observation width"):
            encoder(values[:, :-1])
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("preset,actor_head,critic_encoder", [
    ("cnn1d_base", "mlp", "identity"), ("cnn1d_base", "mlp", "cnn"),
    ("rescnn1d_base", "resmlp", "identity"), ("rescnn1d_base", "mlp", "cnn"),
])
def test_independent_encoder_and_head_choices(preset, actor_head, critic_encoder):
    overrides = [f"network.actor.architecture={actor_head}"]
    if actor_head == "resmlp":
        overrides += ["++network.actor.width=16", "++network.actor.num_blocks=2"]
    if critic_encoder == "cnn":
        overrides += ["network.critic.encoder=${network.actor.encoder}"]
    config = scenario("ppo", preset, small=True, overrides=overrides)
    validate_experiment_scenario(config)
    comp = composer(config)
    resolved = resolve_network_config({"network": config["network"]}, default_hidden_dims=[])
    actor = build_actor(comp.obs_dim, 2, resolved, observation_contract=comp.contract)
    critic = build_critic(comp.obs_dim, resolved, observation_contract=comp.contract)
    x = torch.randn(4, comp.obs_dim)
    loss = actor(x)[0].square().mean() + critic(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in actor.encoder.parameters())
    if critic_encoder == "cnn":
        assert all(p.grad is not None for p in critic.encoder.parameters())


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_native_cpu_training_evaluation_and_recovery(preset, algorithm, tmp_path):
    config = scenario(algorithm, preset, small=True)
    # Also exercise a convolutional PPO critic and shared MAPPO actor routing.
    if algorithm == "ppo":
        config["network"]["critic"]["encoder"] = deepcopy(config["network"]["actor"]["encoder"])
    else:
        config["mappo"]["actor_mode"] = "shared"
        config["network"]["actor"]["parameter_sharing"] = "shared"
    validate_experiment_scenario(config)

    class Stop(Capture):
        def on_update(self, metrics):
            super().on_update(metrics)
            self.should_stop = True

    def construct(stop=False):
        task = create_race_task(config)
        params = learner_params(config, task.spec, algorithm)
        learner = create_learner(algorithm, task.spec, params)
        hook = Stop(False) if stop else Capture(False)
        return OnPolicyTrainer(task, learner, hooks=[hook], scenario=config), hook

    original, capture = construct(True)
    checkpoint = CheckpointHook(original.agent, tmp_path, save_every_steps=1)
    original.hooks.append(checkpoint)
    checkpoint.bind_trainer(original)
    before = deepcopy(original.agent.actor.state_dict())
    original.train(total_steps=7)
    assert 0 < original.collected_steps < 7
    assert all(math.isfinite(value) for row in capture.updates for value in row.values()
               if isinstance(value, (int, float)))
    assert any(not torch.equal(value, before[key]) for key, value in original.agent.actor.state_dict().items()
               if key.startswith("encoder."))
    payload = read_checkpoint(tmp_path / "latest.pt")
    assert payload["schema_version"] == 1
    assert payload["metadata"]["structure"]["encoder"]["actor"] == model_spec(original.agent.network_config, "actor")["encoder"]["type"]
    assert payload["metadata"]["structure"]["observation_segments"]["actor"]
    recovered, _ = construct()
    try:
        recovered.agent.load_for_evaluation(tmp_path / "latest.pt")
        for key, value in original.agent.actor.state_dict().items():
            torch.testing.assert_close(value, recovered.agent.actor.state_dict()[key], rtol=0, atol=0)
        state, _ = recovered.agent.resume_training(tmp_path / "latest.pt")
        assert recovered.agent.optimizer.state
        for key, value in original.agent.critic.state_dict().items():
            torch.testing.assert_close(value, recovered.agent.critic.state_dict()[key], rtol=0, atol=0)
        recovered.restore_training_state(state)
        restore_rng(state["rng"])
        recovered.train(total_steps=7)
        assert recovered.collected_steps == 7
    finally:
        original.task.close()
        recovered.task.close()


@pytest.mark.parametrize("field", ["encoder", "kernel", "segments", "contract", "tensor"])
def test_checkpoint_mismatches_reject_before_mutation(field, tmp_path):
    config = scenario(small=True)
    task = create_race_task(config)
    try:
        params = learner_params(config, task.spec, "ppo")
        source = create_learner("ppo", task.spec, params, training=False)
        path = tmp_path / "cnn.pt"
        source.save_checkpoint(path)
        payload = read_checkpoint(path)
        spec = payload["metadata"]["network"]["actor"]["encoder"]
        if field == "encoder":
            spec["type"] = "rescnn1d_fusion"
        elif field == "kernel":
            spec["kernel_size"] = 3
        elif field == "segments":
            payload["metadata"]["structure"]["observation_segments"]["actor"]["car_0"]["lidar"][1] += 1
        elif field == "contract":
            payload["metadata"]["contracts"]["observations"]["car_0"]["contract"]["observation"]["lidar"]["normalize"] = False
        else:
            payload["models"]["actor"]["encoder.lidar_backbone.0.weight"] = torch.zeros(1)
        atomic_save(payload, path)
        target = create_learner("ppo", task.spec, params, training=False)
        before = deepcopy(target.actor.state_dict())
        with pytest.raises(ValueError, match="Incompatible"):
            target.load_for_evaluation(path)
        for key, value in before.items():
            torch.testing.assert_close(value, target.actor.state_dict()[key], rtol=0, atol=0)
    finally:
        task.close()


@pytest.mark.parametrize("preset", PRESETS)
def test_lora_global_state_and_transfer_boundaries(preset, tmp_path):
    config = scenario("mappo", preset, small=True, overrides=["adaptation=lora"])
    with pytest.raises(ScenarioError, match="CNN encoders.*LoRA"):
        validate_experiment_scenario(config)
    config = scenario("mappo", preset, small=True, overrides=["network.critic.encoder=${network.actor.encoder}"])
    with pytest.raises(ScenarioError, match="global-state critic"):
        validate_experiment_scenario(config)
    tasks = [create_race_task(scenario(algorithm, preset, small=True)) for algorithm in ("ppo", "mappo")]
    try:
        source, target = [create_learner(algorithm, task.spec,
            learner_params(scenario(algorithm, preset, small=True), task.spec, algorithm), training=False)
            for algorithm, task in zip(("ppo", "mappo"), tasks)]
        with pytest.raises(ValueError, match="CNN encoders.*LoRA"):
            LoRAActor(source.actor, {"rank": 1}, 1)
        with pytest.raises(ValueError, match="global-state inputs"):
            build_critic(target.critic_input_dim, source.network_config | {
                "critic": source.network_config["actor"]})
        path = tmp_path / "source.pt"
        source.save_checkpoint(path)
        identity_config = scenario("ppo", preset, small=True)
        identity_config["network"]["actor"]["encoder"] = "identity"
        identity = create_learner("ppo", tasks[0].spec,
            learner_params(identity_config, tasks[0].spec, "ppo"), training=False)
        with pytest.raises(ValueError, match="Incompatible"):
            identity.initialize_from_checkpoint(path, scope="actor_only")
        before = deepcopy(target.actor.state_dict())
        with pytest.raises(ValueError, match="Phase 4"):
            target.initialize_from_checkpoint(path, scope="actor_only", observation_extension="frenet_neighbors")
        for key, value in before.items():
            torch.testing.assert_close(value, target.actor.state_dict()[key], rtol=0, atol=0)
    finally:
        for task in tasks:
            task.close()


@pytest.mark.parametrize("change", ["dimension", "beams", "points", "neighbors", "component", "version"])
def test_inconsistent_observation_segments_rejected(change):
    comp = composer(scenario("mappo", small=True))
    contract = deepcopy(comp.contract)
    width = comp.obs_dim
    if change == "dimension":
        width += 1
    elif change == "beams":
        contract["lidar_beams"] = 0
    elif change == "points":
        contract["observation"]["frenet_vehicle_track"]["points"] = -1
    elif change == "neighbors":
        contract["observation"]["frenet_neighbors"]["agent_ids"] = ["duplicate", "duplicate"]
    elif change == "component":
        contract["observation"]["unknown"] = {"enabled": True}
    else:
        contract["version"] = 99
    with pytest.raises(ValueError):
        observation_segments(contract, width)


@pytest.mark.parametrize("settings", [{"channels": []}, {"kernel_size": 4}, {"stride": True},
    {"lidar_features": 0}, {"num_blocks": 0}, {"normalization": "batchnorm"}])
def test_invalid_encoder_settings_rejected(settings):
    with pytest.raises(ValueError):
        resolve_encoder_config({"type": "rescnn1d_fusion", **settings})


def test_residual_shortcuts_and_shape_validation():
    projected = ResidualConv1DBlock(2, 4, kernel_size=5, stride=2)
    x = torch.randn(3, 2, 17, requires_grad=True)
    with torch.no_grad():
        for parameter in projected.branch.parameters():
            parameter.zero_()
    torch.testing.assert_close(projected(x), projected.shortcut(x), rtol=0, atol=0)
    assert projected(x).shape == (3, 4, 9)
    assert isinstance(ResidualConv1DBlock(2, 2).shortcut, nn.Identity)
    with pytest.raises(ValueError, match="Residual CNN requires"):
        projected(torch.randn(3, 4, 17))
    with pytest.raises(ValueError, match="must be odd"):
        ResidualConv1DBlock(2, 4, kernel_size=4)


def test_representative_process_collector_copies_and_updates_cnn_weights():
    config = scenario(small=True, overrides=["experiment.parallel_envs=2", "experiment.envs_per_worker=1"])
    task = create_race_task(config)
    learner = create_learner("ppo", task.spec, learner_params(config, task.spec, "ppo"))
    before = deepcopy(learner.actor.state_dict())
    capture = Capture(False)
    training = OnPolicyTrainer(task, learner, hooks=[capture], scenario=config, parallel_envs=2,
                               collection_device="cpu")
    existing = {child.pid for child in multiprocessing.active_children()}
    try:
        training.train(total_steps=8)
        assert training.collected_steps == 8
        assert len(capture.updates) >= 2
        assert any(not torch.equal(value, before[key]) for key, value in learner.actor.state_dict().items()
                   if key.startswith("encoder."))
        assert {child.pid for child in multiprocessing.active_children()} <= existing
    finally:
        task.close()
