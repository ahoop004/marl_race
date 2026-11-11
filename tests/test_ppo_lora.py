"""Solo PPO adapters preserve the base, train, and survive native recovery."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.common.checkpoints import read_checkpoint, solo_evaluation_actor
from agents.common.ppo_policy import PPOPolicy
from core.scenario import load_and_expand_scenario
from application.configuration import validate_experiment_scenario
from training.hooks import CheckpointHook
from test_torchrl_ppo import agent, rollout, trainer, Capture


LORA = {"mode": "shared", "rank": 1, "alpha": 1., "train_log_std": True}


def initialized(tmp_path, **options):
    source = agent(hidden_dims=[8])
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    learner = agent(hidden_dims=[8], lora={**LORA, **options})
    learner.initialize_from_checkpoint(path, scope="actor_and_critic")
    return source, learner


@pytest.mark.parametrize("train_log_std", [False, True])
def test_ppo_lora_preserves_initial_policy_and_freezes_base_during_updates(tmp_path, train_log_std):
    torch.manual_seed(42)
    source, learner = initialized(tmp_path, train_log_std=train_log_std)
    obs = np.array([[0.], [1.], [2.], [3.]], dtype=np.float32)
    with torch.no_grad():
        expected = source.actor(torch.as_tensor(obs))
        actual = learner.actor(torch.as_tensor(obs))
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in source.critic.state_dict().items():
        torch.testing.assert_close(learner.critic.state_dict()[key], value, rtol=0, atol=0)
    before = {key: value.clone() for key, value in learner.actor.state_dict().items()}
    critic = deepcopy(learner.critic.state_dict())
    metrics = learner.update(rollout(learner.policy, obs))
    assert metrics["train/optimizer_steps"] > 0
    after = learner.actor.state_dict()
    for key, value in before.items():
        if key.startswith(("net.", "encoder.")) or (key == "log_std" and not train_log_std):
            torch.testing.assert_close(after[key], value, rtol=0, atol=0)
    assert any(not torch.equal(after[k], v) for k, v in before.items() if k.startswith("adapters."))
    assert any(not torch.equal(learner.critic.state_dict()[k], v) for k, v in critic.items())
    assert (not torch.equal(before["log_std"], after["log_std"])) == train_log_std
    optimized = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in (*learner.actor.parameters(), *learner.critic.parameters()) if p.requires_grad}
    assert all(not p.requires_grad for p in learner.actor.net.parameters())


def test_ppo_lora_requires_source_and_shared_mode(tmp_path):
    learner = agent(hidden_dims=[8], lora=LORA)
    for operation in (lambda: learner.predict(np.array([0.])),
                      lambda: learner.sample_batch(np.zeros((1, 1))),
                      lambda: learner.save_checkpoint(tmp_path / "bad.pt")):
        with pytest.raises(ValueError, match="pretrained PPO"):
            operation()
    with pytest.raises(ValueError, match="mode=shared"):
        agent(hidden_dims=[8], lora={**LORA, "mode": "per_agent"})


def test_ppo_lora_evaluation_and_actor_only_extraction_roundtrip(tmp_path):
    _, learner = initialized(tmp_path)
    learner.update(rollout(learner.policy, [0, 1, 2, 3]))
    path = tmp_path / "adapter.pt"
    learner.save_checkpoint(path)
    (tmp_path / "source.pt").unlink()  # Saved adapters carry their frozen base.
    restored = PPOPolicy(1, learner.action_low, learner.action_high,
                         {"hidden_dims": [8], "device": "cpu", "lora": LORA})
    restored.load_for_evaluation(path)
    observation = np.array([0.3], dtype=np.float32)
    np.testing.assert_array_equal(restored.predict(observation), learner.predict(observation))
    checkpoint = read_checkpoint(path)
    spec = SimpleNamespace(agent_ids=["policy"], observation_dims={"policy": 1},
        observation_contracts={"policy": None}, action_lows={"policy": learner.action_low},
        action_highs={"policy": learner.action_high})
    extracted = solo_evaluation_actor(checkpoint, spec, {
        "network": learner.network_config, "device": "cpu", "_action_contract": learner.action_contract,
        "_physics_contract": learner.physics_contract}, "policy")
    np.testing.assert_array_equal(extracted.evaluation_actions(["policy"], {"policy": observation})["policy"],
                                  learner.predict(observation))
    assert extracted.identity["adapter_active"] and not extracted.identity["critic_loaded"]
    bad = deepcopy(checkpoint)
    bad["metadata"]["adaptation"]["source_checkpoint"]["scope"] = "actor_only"
    with pytest.raises(ValueError, match="source identity"):
        restored.load_for_evaluation(bad)
    with pytest.raises(ValueError, match="source identity"):
        solo_evaluation_actor(bad, spec, {"network": learner.network_config, "device": "cpu",
            "_action_contract": learner.action_contract, "_physics_contract": learner.physics_contract}, "policy")


def test_ppo_lora_resume_restores_optimizer_scheduler_and_continues(tmp_path):
    source = agent(hidden_dims=[8])
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    options = dict(hidden_dims=[8], lora=LORA, lr_schedule="step", lr_scheduler_kwargs={"step_size": 1, "gamma": .8})
    class Stop(Capture):
        def on_update(self, metrics):
            super().on_update(metrics)
            self.should_stop = True

    training, _ = trainer(Stop(False), **options)
    training.agent.initialize_from_checkpoint(path, scope="actor_and_critic")
    training.hooks.append(CheckpointHook(training.agent, tmp_path / "run", save_every_steps=4))
    training.train(total_steps=8)
    recovered, _ = trainer(Capture(False), **options)
    state, _ = recovered.agent.resume_training(tmp_path / "run" / "latest.pt")
    recovered.restore_training_state(state)
    np.testing.assert_array_equal(recovered.agent.predict(np.array([.3])), training.agent.predict(np.array([.3])))
    assert recovered.agent.optimizer.state_dict()["param_groups"] == training.agent.optimizer.state_dict()["param_groups"]
    for key, values in training.agent.optimizer.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(recovered.agent.optimizer.state_dict()["state"][key][name], value)
    assert recovered.agent.scheduler.state_dict() == training.agent.scheduler.state_dict()
    recovered.train(total_steps=8)
    assert recovered.collected_steps == 8
    assert all(not p.requires_grad for p in recovered.agent.actor.net.parameters())


@pytest.mark.parametrize("family", [None, "straight", "sweeper", "hairpin", "chicane", "s_curve"])
def test_generated_ppo_scenarios_preserve_splits_and_adapter_params(family):
    overrides = [] if family is None else [f"maps=generated_{family}"]
    config = load_and_expand_scenario("scenarios/ppo_generated_lora.yaml", overrides=overrides)
    validate_experiment_scenario(config)
    assert config["training_defaults"]["lora"]["mode"] == "shared"
    maps = config["maps"]
    assert [len(maps[k]) for k in ("train", "selection", "final_test")] == ([40, 8, 12] if family is None else [4, 1, 1])
    train, validation, test = (set(maps[key]) for key in ("train", "selection", "final_test"))
    assert train.isdisjoint(validation) and train.isdisjoint(test) and validation.isdisjoint(test)
    from core.map_selection import resolve_bundle_yaml
    from pathlib import Path
    assert all(resolve_bundle_yaml(Path(maps["root"]), p).is_file() for p in maps["available"])
    full = load_and_expand_scenario("scenarios/ppo_generated_finetune.yaml", overrides=overrides)
    validate_experiment_scenario(full)
    assert full["environment"] == config["environment"]
    assert full["agents"] == config["agents"]
    assert full["training_defaults"].get("lora") is None


def test_cli_checkpoint_override_retains_ppo_lora():
    from application.cli import apply_cli_overrides, parse_args
    path = "scenarios/ppo_generated_lora.yaml"
    args = parse_args(["--scenario", path, "--checkpoint", "/tmp/source.pt"])
    config = apply_cli_overrides(load_and_expand_scenario(path), args)
    validate_experiment_scenario(config)
    assert config["adaptation"]["mode"] == "lora"
    assert config["adaptation"]["checkpoint"] == "/tmp/source.pt"


@pytest.mark.parametrize("field", ["physics", "observations", "actions", "rank"])
def test_ppo_lora_rejects_incompatible_checkpoint_before_mutation(tmp_path, field):
    _, learner = initialized(tmp_path)
    path = tmp_path / "adapter.pt"
    learner.save_checkpoint(path)
    bad = read_checkpoint(path)
    if field == "rank":
        bad["metadata"]["adaptation"]["lora"]["rank"] += 1
    else:
        bad["metadata"]["contracts"][field] = {"changed": True}
    before = deepcopy(learner.actor.state_dict())
    with pytest.raises(ValueError, match="Incompatible"):
        learner.load_for_evaluation(bad)
    for key, value in before.items():
        torch.testing.assert_close(learner.actor.state_dict()[key], value, rtol=0, atol=0)


def test_ppo_lora_transfer_rejects_actor_only_scope(tmp_path):
    learner = agent(hidden_dims=[8], lora=LORA)
    with pytest.raises(ValueError, match="actor_and_critic"):
        learner.initialize_from_checkpoint(tmp_path / "unused.pt", scope="actor_only")
