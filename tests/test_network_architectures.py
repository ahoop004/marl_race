"""Focused architecture, historical-weight and native recovery contracts."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch
from torch import nn

from agents.common.checkpoints import read_checkpoint, restore_rng
from agents.common.encoders import build_encoder
from agents.common.mappo_policy import MAPPOPolicy
from agents.common.networks import (
    ResidualDenseBlock, build_actor, build_critic, model_spec, resolve_network_config,
)
from agents.common.ppo_policy import PPOPolicy
from application.configuration import validate_experiment_scenario
from core.scenario import load_and_expand_scenario, ScenarioError
from core.task_builder import create_race_task
from training.algorithms import create_learner, learner_params
from training.hooks import CheckpointHook
from training.on_policy import OnPolicyTrainer
from test_native_torchrl_env import team_task
from test_torchrl_ppo import Capture, trainer as ppo_trainer
from test_torchrl_mappo import BOUNDS, IDS, params as mappo_params
from agents.torchrl_mappo import TorchRLMAPPOAgent
from utils.torch_io import atomic_save


PRETRAIN = "scenarios/ppo_lap_completion_pretrain.yaml"
TEAM = "scenarios/mappo_2v2_completion_scratch.yaml"
PRESETS = {"mlp": [256, 256], "mlp_small": [128, 128], "mlp_base": [256, 256],
           "mlp_wide": [512, 512], "mlp_deep": [256] * 4, "resmlp_base": None}


def network(actor="mlp", critic="mlp", **extra):
    return dict(architecture="mlp", actor_hidden_dims=[8, 4], critic_hidden_dims=[12],
                activation="leaky_relu", actor={"architecture": actor, **(
                    {"width": 8, "num_blocks": 2} if actor == "resmlp" else {})},
                critic={"architecture": critic, **(
                    {"width": 12, "num_blocks": 2} if critic == "resmlp" else {})}, **extra)


def test_historical_seeded_layers_saved_weights_and_metadata_are_identical(tmp_path):
    # Independent historical construction: actor, log_std=zeros, then critic.
    torch.manual_seed(71)
    actor_net = nn.Sequential(nn.Linear(7, 8), nn.LeakyReLU(.2), nn.Linear(8, 4),
                              nn.LeakyReLU(.2), nn.Linear(4, 2))
    critic_net = nn.Sequential(nn.Linear(7, 12), nn.LeakyReLU(.2), nn.Linear(12, 1))
    legacy_rng = torch.random.get_rng_state().clone()
    config = network()
    torch.manual_seed(71)
    policy = PPOPolicy(7, *BOUNDS, {"network": config, "device": "cpu"})
    assert torch.equal(torch.random.get_rng_state(), legacy_rng)
    assert policy.network_config == dict(architecture="mlp", actor_hidden_dims=[8, 4],
                                         critic_hidden_dims=[12], activation="leaky_relu")
    actor_state = {"log_std": torch.zeros(2), **{f"net.{k}": v for k, v in actor_net.state_dict().items()}}
    critic_state = {f"net.{k}": v for k, v in critic_net.state_dict().items()}
    for model, state in ((policy.actor, actor_state), (policy.critic, critic_state)):
        assert set(model.state_dict()) == set(state)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    path = tmp_path / "legacy.pt"
    policy.save_checkpoint(path)
    payload = read_checkpoint(path)
    assert payload["schema_version"] == 1
    assert payload["metadata"]["network"] == policy.network_config
    payload["models"] = {"actor": actor_state, "critic": critic_state}
    atomic_save(payload, path)
    restored = PPOPolicy(7, *BOUNDS, {"network": config, "device": "cpu"})
    restored.load_for_evaluation(path)
    restored.critic.load_state_dict(critic_state, strict=True)
    x = torch.linspace(-2, 2, 35).reshape(5, 7)
    mean, std = restored.actor(x)
    torch.testing.assert_close(mean, actor_net(x), rtol=0, atol=0)
    torch.testing.assert_close(std, torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(restored.critic(x), critic_net(x).squeeze(-1), rtol=0, atol=0)
    np.testing.assert_array_equal(restored.predict(x[0].numpy()), actor_net(x[:1]).tanh()[0].detach().numpy())


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
@pytest.mark.parametrize("beams", [108, 1080])
def test_hydra_presets_real_observation_specs_and_factory_shapes(preset, algorithm, beams):
    config = load_and_expand_scenario(PRETRAIN if algorithm == "ppo" else TEAM, overrides=[
        f"network={preset}", f"environment.lidar_beams={beams}", "execution.params.device=cpu"])
    validate_experiment_scenario(config)
    task = create_race_task(config)
    try:
        params = learner_params(config, task.spec, algorithm)
        torch.manual_seed(123)
        policy = create_learner(algorithm, task.spec, params, training=False)
        first = deepcopy(policy.actor.state_dict())
        torch.manual_seed(123)
        repeated = create_learner(algorithm, task.spec, params, training=False)
        for key, value in repeated.actor.state_dict().items():
            torch.testing.assert_close(value, first[key], rtol=0, atol=0)
        role = model_spec(policy.network_config, "actor")
        if PRESETS[preset] is not None:
            assert role["hidden_dims"] == PRESETS[preset]
            assert model_spec(policy.network_config, "critic")["hidden_dims"] == [512, 512]
        else:
            assert (role["width"], role["num_blocks"], role["input_projection"]) == (256, 4, True)
        dim = task.spec.observation_dims[task.spec.agent_ids[0]]
        assert dim == beams + (50 if algorithm == "ppo" else 84)
        # The nested leading dimensions are preserved by encoder and head.
        actor = (policy.actor.actors[task.spec.agent_ids[0]]
                 if getattr(policy, "actor_mode", None) == "independent" and algorithm == "mappo"
                 else policy.actor)
        mean, std = actor(torch.zeros(2, 3, dim))
        assert mean.shape == (2, 3, 2) and std.shape == (2,)
        critic_dim = dim if algorithm == "ppo" else policy.critic_input_dim
        assert policy.critic(torch.zeros(2, 3, critic_dim)).shape == (2, 3)
        assert actor.encoder.output_dim == dim
    finally:
        task.close()


@pytest.mark.parametrize("actor", ["mlp", "resmlp"])
@pytest.mark.parametrize("critic", ["mlp", "resmlp"])
@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "tanh", "silu"])
def test_independent_heads_activations_and_centralized_critic_inputs(actor, critic, activation):
    raw = network(actor, critic)
    raw["actor"]["activation"] = activation
    raw["critic"]["activation"] = "tanh"
    for mode, width in (("shared_team", 17), ("agent_conditioned", 19)):
        policy = MAPPOPolicy(7, 17, *BOUNDS, IDS, mappo_params(network=raw, critic_mode=mode, team_return_mode="per_agent"))
        assert policy.actor(torch.zeros(3, 7))[0].shape == (3, 2)
        assert policy.critic(torch.zeros(3, width)).shape == (3,)
        assert model_spec(policy.network_config, "actor")["activation"] == activation
        assert model_spec(policy.network_config, "critic")["activation"] == "tanh"
        assert resolve_network_config({"network": policy.network_config}, default_hidden_dims=[]) == policy.network_config


@pytest.mark.parametrize("actor,critic", [("resmlp", "mlp"), ("mlp", "resmlp"), ("resmlp", "resmlp")])
@pytest.mark.parametrize("algorithm", ["ppo", "mappo_shared", "mappo_independent", "mappo_lora"])
def test_native_cpu_updates_checkpoint_evaluation_and_resume(actor, critic, algorithm, tmp_path):
    raw = network(actor, critic)
    if algorithm == "mappo_lora":
        PPOPolicy(1, *BOUNDS, {"network": raw}).save_checkpoint(tmp_path / "source.pt")

    class Stop(Capture):
        def on_update(self, metrics):
            super().on_update(metrics)
            self.should_stop = True

    def construct(stop=False):
        hook = Stop(False) if stop else Capture(False)
        if algorithm == "ppo":
            return ppo_trainer(hook, network=raw, n_epochs=1)
        task = team_task({3: ((), True, (), (*IDS, "car_2", "car_3"))})
        options = mappo_params(network=raw, n_steps=8, actor_mode=algorithm.removeprefix("mappo_"))
        if algorithm == "mappo_lora":
            options.update(actor_mode="shared", lora={"mode": "per_agent", "rank": 1, "per_agent_log_std": True})
        policy = TorchRLMAPPOAgent(1, task.state_space().shape[0], *BOUNDS, IDS, options)
        if stop and algorithm == "mappo_lora":
            policy.initialize_from_checkpoint(tmp_path / "source.pt", scope="actor_only")
        return OnPolicyTrainer(task, policy, hooks=[hook]), None

    original, _ = construct(stop=True)
    checkpoint = CheckpointHook(original.agent, tmp_path, save_every_steps=4)
    original.hooks.append(checkpoint)
    checkpoint.bind_trainer(original)
    original.train(total_steps=7)
    assert original.collected_steps == 4
    saved = read_checkpoint(tmp_path / "latest.pt")
    assert saved["schema_version"] == 1
    assert saved["metadata"]["network"]["spec_version"] == 1
    assert original.agent.optimizer.state
    for metrics in original.hooks[0].updates:
        assert all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
    recovered, _ = construct()
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
    assert recovered.hooks[0].updates


@pytest.mark.parametrize("change", ["architecture", "blocks", "activation", "projection", "missing_version"])
def test_incompatible_architectures_reject_before_loading_weights(change, tmp_path):
    raw = network("resmlp", "resmlp")
    source = PPOPolicy(7, *BOUNDS, {"network": raw})
    path = tmp_path / "residual.pt"
    source.save_checkpoint(path)
    other = deepcopy(raw)
    if change == "architecture":
        other = network()
    elif change == "blocks":
        other["actor"]["num_blocks"] = 3
    elif change == "activation":
        other["critic"]["activation"] = "relu"
    elif change == "projection":
        # Same shape, different computation: metadata must still reject.
        other["actor"]["width"] = 7
        raw["actor"]["width"] = 7
        source = PPOPolicy(7, *BOUNDS, {"network": raw})
        source.save_checkpoint(path)
        other["actor"]["input_projection"] = False
    else:
        saved = read_checkpoint(path)
        saved["metadata"]["network"].pop("spec_version")
        atomic_save(saved, path)
    target = PPOPolicy(7, *BOUNDS, {"network": other})
    before = deepcopy(target.actor.state_dict())
    with pytest.raises(ValueError, match="Incompatible"):
        target.load_for_evaluation(path)
    for key, value in before.items():
        torch.testing.assert_close(value, target.actor.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("actor,critic", [("resmlp", "mlp"), ("mlp", "resmlp")])
def test_residual_lora_and_ppo_to_mappo_transfer_supported(actor, critic, tmp_path):
    raw = network(actor, critic)
    adapted = MAPPOPolicy(7, 17, *BOUNDS, IDS, mappo_params(network=raw, lora={"rank": 1}))
    assert all(not p.requires_grad for p in adapted.actor.net.parameters())
    source = PPOPolicy(7, *BOUNDS, {"network": raw})
    source.save_checkpoint(tmp_path / "ppo.pt")
    target = MAPPOPolicy(7, 17, *BOUNDS, IDS, mappo_params(network=raw))
    critic_before = deepcopy(target.critic.state_dict())
    target.initialize_from_checkpoint(tmp_path / "ppo.pt", scope="actor_only")
    x = torch.randn(5, 7)
    torch.testing.assert_close(target.actor(x)[0], source.actor(x)[0], rtol=0, atol=0)
    for key, value in critic_before.items():
        torch.testing.assert_close(value, target.critic.state_dict()[key], rtol=0, atol=0)
    config = load_and_expand_scenario(TEAM, overrides=["network=resmlp_base", "adaptation=lora"])
    validate_experiment_scenario(config)


@pytest.mark.parametrize("override,message", [
    ({"actor": {"encoder": "cnn1d"}}, "Unsupported observation encoder"),
    ({"actor": {"width": 8}}, "require resmlp"),
    ({"actor": {"architecture": "resmlp", "hidden_dims": [8]}}, "uses width"),
    ({"critic": {"architecture": "resmlp", "num_blocks": 0}}, "positive integer"),
    ({"actor": {"activation": "gelu"}}, "Unsupported network activation"),
    ({"actor": {"typo": 1}}, "Unknown network.actor"),
    ({"actor": {"observation_layout": "image"}}, "observation_layout"),
])
def test_invalid_model_specs_fail_clearly(override, message):
    with pytest.raises(ValueError, match=message):
        resolve_network_config({"network": {**network(), **override}}, default_hidden_dims=[])


def test_encoder_residual_shape_validation_and_optional_projection():
    with pytest.raises(ValueError, match="positive integer"):
        build_encoder(0)
    with pytest.raises(ValueError, match="observation width"):
        build_encoder(3)(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="last dimension"):
        ResidualDenseBlock(3)(torch.zeros(2, 4))
    raw = network("resmlp")
    raw["actor"]["input_projection"] = False
    resolved = resolve_network_config({"network": raw}, default_hidden_dims=[])
    with pytest.raises(ValueError, match="input_projection=true"):
        build_actor(7, 2, resolved)
    assert build_actor(8, 2, resolved)(torch.zeros(5, 8))[0].shape == (5, 2)
    assert build_critic(7, resolved)(torch.zeros(5, 7)).shape == (5,)



@pytest.mark.parametrize("preset", ["mlp_base", "resmlp_base", "cnn1d_base", "rescnn1d_base"])
@pytest.mark.parametrize("adaptation", ["scratch", "full_finetune"])
def test_modular_presets_reuse_adaptation_configuration(preset, adaptation, tmp_path):
    from training.algorithms import resolve_training_params

    overrides = [f"network={preset}", f"adaptation={adaptation}", "execution.params.device=cpu"]
    if adaptation == "full_finetune":
        overrides.append(f"adaptation.checkpoint={tmp_path / 'source.pt'}")
    config = load_and_expand_scenario(TEAM, overrides=overrides)
    validate_experiment_scenario(config)
    params = resolve_training_params(config["agents"]["car_0"], config)
    assert params["require_pretrained_actor"] == (adaptation == "full_finetune")
    assert config["adaptation"]["load_scope"] == "actor_only"
    assert params["pretrained_actor_observation_extension"] == "frenet_neighbors"
    assert params["pretrained_actor_checkpoint"] == (str(tmp_path / "source.pt") if adaptation == "full_finetune" else None)


def test_residual_transfer_requires_an_input_projection():
    from agents.common.networks import expand_observation_inputs

    resolved = resolve_network_config({"network": network("resmlp")}, default_hidden_dims=[])
    resolved["actor"]["input_projection"] = False
    with pytest.raises(ValueError, match="input projection"):
        expand_observation_inputs({}, {}, resolved, source_width=7, expand_inputs=True)


@pytest.mark.parametrize("preset", ["mlp_base", "resmlp_base", "cnn1d_base", "rescnn1d_base"])
@pytest.mark.parametrize("bad", ["target", "rank", "no_targets"])
def test_modular_lora_configuration_validates_targets_and_every_layer(preset, bad):
    overrides = [f"network={preset}", "adaptation=lora", "execution.params.device=cpu"]
    if bad == "target":
        overrides.append("adaptation.lora.target=encoder")
    elif bad == "rank":
        overrides.append("adaptation.lora.rank=1000")
    else:
        overrides += ["network.actor.architecture=mlp", "network.actor_hidden_dims=[]"]
        if preset == "resmlp_base":
            # Residual-only options are intentionally not legal for an MLP head.
            overrides += ["~network.actor.width", "~network.actor.num_blocks", "~network.actor.input_projection"]
    config = load_and_expand_scenario(TEAM, overrides=overrides)
    with pytest.raises(ScenarioError, match="lora.target|lora.rank|hidden layer"):
        validate_experiment_scenario(config)


def test_lora_rank_validation_uses_encoded_feature_dimensions():
    config = load_and_expand_scenario(TEAM, overrides=["network=cnn1d_base", "adaptation=lora",
        "network.actor.encoder.lidar_features=2", "network.actor.encoder.vehicle_features=2",
        "network.actor.encoder.neighbor_features=2", "network.actor_hidden_dims=[8]", "adaptation.lora.rank=7"])
    with pytest.raises(ScenarioError, match="net.0.*dimensions"):
        validate_experiment_scenario(config)
