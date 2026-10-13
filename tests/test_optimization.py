"""Numerical parity, configuration, schedule clocks and recovery contracts."""
from copy import deepcopy
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.utils import distance_loss

from agents.common.checkpoints import optimization_contract, read_checkpoint
from agents.common.optimization import OPTIMIZERS
from application.checkpoints import validate_resume_experiment
from application.configuration import validate_experiment_scenario
from core.provenance import collect_map_protocols
from core.scenario import ScenarioError, load_and_expand_scenario
from loggers.wandb_logger import WandbLogger
from test_torchrl_ppo import agent, rollout, trainer, Capture
from test_torchrl_mappo import learner, transition, IDS
from training.algorithms import resolve_training_params
from training.hooks import CheckpointHook, WandbHook


def batch(policy):
    if policy.algorithm == "ppo":
        if policy.device.type == "cpu":
            return rollout(policy.policy, [0, 1, 2, 3])
        data = TensorDict({"observation": torch.arange(4, dtype=torch.float32, device=policy.device).reshape(4, 1)}, [4])
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            policy.policy(data)
        data = data.cpu()
        data["next"] = TensorDict({"observation": data["observation"] + 1, "reward": torch.ones(4, 1),
                                   "done": torch.zeros(4, 1, dtype=torch.bool),
                                   "terminated": torch.zeros(4, 1, dtype=torch.bool)}, [4])
        return data
    return torch.stack([transition(policy, IDS, i, i + 1, terminal=i == 1) for i in range(2)])


@pytest.mark.parametrize("factory", [agent, learner], ids=["ppo", "mappo"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("optimizer,kwargs", [
    ("adam", {"betas": [0.8, 0.95], "eps": 1e-7, "weight_decay": 0.01, "amsgrad": True}),
    ("adamw", {"betas": [0.8, 0.95], "weight_decay": 0.02}),
    ("sgd", {"momentum": 0.9, "nesterov": True, "weight_decay": 0.01}),
    ("rmsprop", {"alpha": 0.95, "momentum": 0.9, "centered": True, "eps": 1e-7}),
])
def test_registered_optimizers_update_both_learners(factory, optimizer, kwargs, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = factory(optimizer=optimizer, optimizer_kwargs=kwargs, lr_schedule="cosine",
                     learning_rate=1e-3, learning_rate_end=1e-4, device=device)
    assert isinstance(policy.optimizer, OPTIMIZERS[optimizer])
    before = [p.detach().clone() for p in policy._optim_parameters]
    policy.set_training_progress(0.5)
    metrics = policy.update(batch(policy))
    assert metrics["train/optimizer_steps"] > 0
    assert metrics["train/learning_rate"] == pytest.approx(5.5e-4)
    assert any(not torch.equal(old, new) for old, new in zip(before, policy._optim_parameters))
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("factory", [agent, learner], ids=["ppo", "mappo"])
@pytest.mark.parametrize("schedule,progress,expected", [
    ("constant", 0.5, 1e-3),
    ("linear", 0, 1e-3), ("linear", 0.25, 7.75e-4), ("linear", 1, 1e-4),
    ("cosine", 0, 1e-3), ("cosine", 0.25, 1e-4 + 9e-4 * (1 + math.cos(math.pi/4))/2),
    ("cosine", 0.5, 5.5e-4), ("cosine", 1, 1e-4), ("cosine", 2, 1e-4),
])
def test_budget_schedules_have_correct_endpoints_and_curve(factory, schedule, progress, expected):
    policy = factory(lr_schedule=schedule, learning_rate=1e-3,
                     learning_rate_end=None if schedule == "constant" else 1e-4)
    policy.set_training_progress(progress)
    assert policy.optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    assert policy.scheduler_clock == "training_budget_progress"
    assert policy.scheduler is None


@pytest.mark.parametrize("factory", [agent, learner], ids=["ppo", "mappo"])
@pytest.mark.parametrize("schedule,kwargs,rates", [
    ("StepLR", {"step_size": 2, "gamma": 0.5}, [1e-3, 1e-3, 5e-4]),
    ("ExponentialLR", {"gamma": 0.5}, [1e-3, 5e-4, 2.5e-4]),
])
def test_native_schedulers_advance_once_per_nonempty_learner_update(factory, schedule, kwargs, rates):
    policy = factory(lr_schedule=schedule, lr_scheduler_kwargs=kwargs, learning_rate=1e-3,
                     n_epochs=2, batch_size=1)
    for i, rate in enumerate(rates):
        policy.set_training_progress(0.9)
        assert policy.optimizer.param_groups[0]["lr"] == pytest.approx(rate)
        metrics = policy.update(batch(policy))
        assert metrics["train/learning_rate"] == pytest.approx(rate)
        assert metrics["train/optimizer_steps"] == 8
        assert policy.scheduler.last_epoch == i + 1
    assert policy.scheduler_clock == "learner_updates"


def test_kl_gate_with_no_optimizer_work_does_not_advance_scheduler():
    policy = agent(lr_schedule="step", lr_scheduler_kwargs={"step_size": 1}, target_kl=1e-6)
    data = batch(policy)
    with torch.no_grad():
        for parameter in policy.actor.parameters():
            parameter.add_(1)
    metrics = policy.update(data)
    assert metrics["train/optimizer_steps"] == 0
    assert policy.scheduler.last_epoch == 0


@pytest.mark.parametrize("factory", [agent, learner], ids=["ppo", "mappo"])
@pytest.mark.parametrize("critic_loss", ["l2", "l1", "smooth_l1"])
@pytest.mark.parametrize("vf_coef", [0, 0.5])
def test_torchrl_value_losses_and_metrics_use_selected_loss(factory, critic_loss, vf_coef):
    policy = factory(critic_loss=critic_loss, vf_coef=vf_coef, ent_coef=0, n_epochs=1, batch_size=4)
    data = batch(policy)
    captured = []
    original = policy.loss_module.forward

    def observe(minibatch):
        key = lambda name: ("agents", name) if policy.algorithm == "mappo" else name
        with torch.no_grad():
            values = policy.value_module(minibatch.clone(False))[key("state_value")]
            expected = distance_loss(values, minibatch[key("value_target")], critic_loss).mean()
        losses = original(minibatch)
        if vf_coef:
            torch.testing.assert_close(losses["loss_critic"], vf_coef * expected)
        captured.append(float(expected))
        return losses

    policy.loss_module.forward = observe
    if not vf_coef:
        # Diagnostic-only value loss is measured after the actor update; the
        # critic receives no gradient and therefore retains these exact values.
        critic_before = deepcopy(policy.critic.state_dict())
    metrics = policy.update(data)
    assert metrics["train/value_loss"] == pytest.approx(sum(captured)/len(captured))
    assert not policy.loss_module.entropy_bonus
    if not vf_coef:
        for key, value in policy.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[key], rtol=0, atol=0)


def test_default_adam_and_l2_update_matches_original_full_minibatch():
    torch.manual_seed(401)
    policy = agent(learning_rate=1e-3, learning_rate_end=1e-4, lr_schedule="linear",
                   n_epochs=1, batch_size=4)
    reference = agent(n_epochs=1, batch_size=4)
    reference.actor.load_state_dict(policy.actor.state_dict())
    reference.critic.load_state_dict(policy.critic.state_dict())
    reference.optimizer = torch.optim.Adam(reference._optim_parameters, lr=7.75e-4)
    reference.loss_module = ClipPPOLoss(
        reference.probabilistic_actor, reference.value_module,
        clip_epsilon=reference.clip_range, entropy_bonus=False, critic_coeff=reference.vf_coef,
        loss_critic_type="l2", normalize_advantage=False, functional=False)
    reference.loss_module.set_keys(action="raw_action", sample_log_prob="raw_log_prob")
    data = batch(policy)
    legacy = data.clone()
    with torch.no_grad():
        reference.gae(legacy)
    advantage = legacy["advantage"]
    legacy["advantage"] = (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
    # One fixed full minibatch: its order does not affect the objective or update.
    losses = reference.loss_module(legacy)
    _, entropy = reference.actor.evaluate_actions(legacy["observation"], legacy["action"], legacy["raw_action"])
    loss = losses["loss_objective"] + losses["loss_critic"] - reference.ent_coef * entropy.mean()
    reference.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference._optim_parameters, reference.max_grad_norm)
    reference.optimizer.step()
    policy.set_training_progress(0.25)
    metrics = policy.update(data)
    assert metrics["train/policy_loss"] == pytest.approx(float(losses["loss_objective"].detach()), abs=1e-6)
    assert metrics["train/value_loss"] == pytest.approx(float(losses["loss_critic"].detach()/reference.vf_coef), abs=1e-6)
    for actual, expected in zip(policy._optim_parameters, reference._optim_parameters):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


class Stop(Capture):
    def on_update(self, metrics):
        super().on_update(metrics)
        self.should_stop = True


@pytest.mark.parametrize("optimizer,schedule,kwargs", [
    ("adam", "cosine", {}), ("adamw", "step", {"step_size": 1, "gamma": 0.8}),
    ("sgd", "exponential", {"gamma": 0.8}), ("rmsprop", "linear", {}),
])
def test_resume_restores_optimizer_lr_and_native_scheduler(tmp_path, optimizer, schedule, kwargs):
    optimizer_kwargs = {"weight_decay": 0.01, **({"momentum": 0.9} if optimizer == "sgd" else {})}
    options = dict(optimizer=optimizer, optimizer_kwargs=optimizer_kwargs,
                   lr_schedule=schedule, lr_scheduler_kwargs=kwargs,
                   learning_rate=1e-3, learning_rate_end=1e-4 if schedule in {"linear", "cosine"} else None)
    original, _ = trainer(Stop(False), **options)
    original.hooks.append(CheckpointHook(original.agent, tmp_path, save_every_steps=4))
    original.train(total_steps=9)
    path = tmp_path / "latest.pt"
    saved = read_checkpoint(path)
    capture = Capture(False)
    recovered, _ = trainer(capture, **options)
    state, _ = recovered.agent.resume_training(path)
    recovered.restore_training_state(state)
    assert optimization_contract(recovered.agent) == optimization_contract(original.agent)
    assert recovered.agent.optimizer.state_dict()["param_groups"] == original.agent.optimizer.state_dict()["param_groups"]
    assert recovered.agent.optimizer.state
    for actual, expected in zip(recovered.agent.optimizer.state.values(), original.agent.optimizer.state.values()):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    if recovered.agent.scheduler is not None:
        assert recovered.agent.scheduler.state_dict() == saved["training"]["scheduler"]
    recovered.train(total_steps=9)
    assert capture.updates[-1]["train/environment_steps"] == 9
    if schedule in {"linear", "cosine"}:
        assert capture.updates[-1]["train/learning_rate"] == pytest.approx(1e-4)
    else:
        assert recovered.agent.scheduler.last_epoch == 3
        assert [m["train/learning_rate"] for m in capture.updates] == pytest.approx([8e-4, 6.4e-4])


@pytest.mark.parametrize("changed", [
    {"optimizer": "adamw"}, {"optimizer_kwargs": {"eps": 1e-6}},
    {"lr_schedule": "cosine"}, {"critic_loss": "l1"},
])
def test_resume_rejects_changed_optimization_before_loading_models(tmp_path, changed):
    options = dict(lr_schedule="linear", learning_rate_end=1e-4)
    original, _ = trainer(Stop(False), **options)
    original.hooks.append(CheckpointHook(original.agent, tmp_path, save_every_steps=4))
    original.train(total_steps=9)
    destination = agent(**{**options, **changed})
    before = deepcopy(destination.actor.state_dict())
    with pytest.raises(ValueError, match="optimization_contract"):
        destination.resume_training(tmp_path / "latest.pt")
    for key, value in destination.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_legacy_contract_and_application_configuration_resume_unambiguously(tmp_path):
    original, _ = trainer(Stop(False), lr_schedule="linear", learning_rate_end=1e-4)
    original.hooks.append(CheckpointHook(original.agent, tmp_path, save_every_steps=4))
    original.train(total_steps=9)
    payload = read_checkpoint(tmp_path / "latest.pt")
    payload["training"].pop("scheduler")
    for key in ("optimizer_name", "optimizer_kwargs", "scheduler_kwargs", "scheduler_clock", "loss", "critic_loss"):
        payload["training"]["optimization_contract"].pop(key)
    current = agent(lr_schedule="linear", learning_rate_end=1e-4)
    current.resume_training(payload)
    assert current.optimizer.param_groups[0]["lr"] == original.agent.optimizer.param_groups[0]["lr"]
    incompatible = agent(lr_schedule="cosine", learning_rate_end=1e-4)
    with pytest.raises(ValueError, match="optimization_contract"):
        incompatible.resume_training(payload)
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml")
    old = deepcopy(scenario)
    old["algorithm"].pop("optimizer_kwargs")
    old["algorithm"]["params"].pop("lr_scheduler_kwargs")
    old["training_defaults"].pop("lr_scheduler_kwargs")
    metadata = {"configuration": old, "provenance": {"map_protocols": collect_map_protocols(scenario["environment"])}}
    validate_resume_experiment(metadata, scenario)


def test_resume_rejects_training_budget_changes_and_partial_contracts(tmp_path):
    original, _ = trainer(Stop(False), lr_schedule="step", lr_scheduler_kwargs={"step_size": 2})
    original.hooks.append(CheckpointHook(original.agent, tmp_path, save_every_steps=4))
    original.train(total_steps=9)
    payload = read_checkpoint(tmp_path / "latest.pt")
    options = dict(lr_schedule="step", lr_scheduler_kwargs={"step_size": 2})
    changed = agent(lr_schedule="step", lr_scheduler_kwargs={"step_size": 3})
    with pytest.raises(ValueError, match="scheduler_kwargs"):
        changed.resume_training(payload)
    recovered, _ = trainer(Capture(False), **options)
    state, _ = recovered.agent.resume_training(payload)
    recovered.restore_training_state(state)
    with pytest.raises(ValueError, match="total_steps"):
        recovered.train(total_steps=10)
    for field, message in (("scheduler", "scheduler recovery"), ("optimizer_name", "incomplete optimization_contract")):
        broken = deepcopy(payload)
        if field == "scheduler":
            broken["training"].pop(field)
        else:
            broken["training"]["optimization_contract"].pop(field)
        with pytest.raises(ValueError, match=message):
            agent(**options).resume_training(broken)


@pytest.mark.parametrize("mode", ["shared", "independent", "lora"])
@pytest.mark.parametrize("schedule", ["cosine", "step"])
def test_mappo_adamw_resume_and_lora_trainable_parameter_ownership(tmp_path, mode, schedule):
    from agents.common.ppo_policy import PPOPolicy
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from test_native_torchrl_env import team_task
    from test_torchrl_mappo import params, BOUNDS
    from training.on_policy import OnPolicyTrainer

    options = params(hidden_dims=[4], n_steps=2, optimizer="adamw", critic_loss="smooth_l1",
                     actor_mode="shared" if mode == "lora" else mode,
                     learning_rate=1e-3, lr_schedule=schedule,
                     learning_rate_end=1e-4 if schedule == "cosine" else None,
                     lr_scheduler_kwargs={"step_size": 1, "gamma": 0.5} if schedule == "step" else {})
    if mode == "lora":
        options["lora"] = {"mode": "per_agent", "rank": 1, "train_log_std": False, "per_agent_log_std": True}
        source = tmp_path / "source.pt"
        PPOPolicy(1, *BOUNDS, {"hidden_dims": [4], "device": "cpu"}).save_checkpoint(source)
    events = {3: ((), True, (), (*IDS, "car_2", "car_3"))}
    task = team_task(events)
    policy = TorchRLMAPPOAgent(1, task.state_space().shape[0], *BOUNDS, IDS, options)
    if mode == "lora":
        policy.initialize_from_checkpoint(source, scope="actor_only")
    frozen = {id(p) for p in policy.actor.parameters() if not p.requires_grad}
    trainable = {id(p) for module in (policy.actor, policy.critic) for p in module.parameters() if p.requires_grad}
    optimized = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert optimized == trainable and not (optimized & frozen)
    if mode == "lora":
        assert frozen
        base = deepcopy(policy.actor.net.state_dict())
    training = OnPolicyTrainer(task, policy, hooks=[Stop(False), CheckpointHook(policy, tmp_path, save_every_steps=2)])
    training.train(total_steps=5)
    new_task = team_task(events)
    recovered = TorchRLMAPPOAgent(1, new_task.state_space().shape[0], *BOUNDS, IDS, options)
    state, _ = recovered.resume_training(tmp_path / "latest.pt")
    assert recovered.optimizer.state_dict()["param_groups"] == policy.optimizer.state_dict()["param_groups"]
    if schedule == "step":
        assert recovered.scheduler.state_dict() == policy.scheduler.state_dict()
    capture = Capture(False)
    continued = OnPolicyTrainer(new_task, recovered, hooks=[capture])
    continued.restore_training_state(state)
    continued.train(total_steps=5)
    assert continued.collected_steps == 5
    if mode == "lora":
        for key, value in recovered.actor.net.state_dict().items():
            torch.testing.assert_close(value, base[key], rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
@pytest.mark.parametrize("collection_backend", ["process", "centralized"])
def test_short_grouped_collector_cosine_training(algorithm, collection_backend):
    from core.configuration import PROJECT_ROOT
    from core.task_builder import create_race_task
    from training.algorithms import create_learner, learner_params
    from training.on_policy import OnPolicyTrainer

    name = "ppo_lap_completion_pretrain" if algorithm == "ppo" else "mappo_2v2_completion_scratch"
    scenario = load_and_expand_scenario(f"scenarios/{name}.yaml", overrides=[
        "execution.params.device=cpu", "experiment.envs_per_worker=2", "experiment.total_steps=8",
        "environment.max_steps=3", "evaluation.enabled=false", "wandb.enabled=false",
        "algorithm.params.n_steps=4", "algorithm.params.n_epochs=1", "algorithm.params.batch_size=4",
        "++algorithm.params.frames_per_batch=4", "algorithm.params.lr_schedule=cosine",
        "++algorithm.params.learning_rate_end=0.0001",
    ])
    validate_experiment_scenario(scenario)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    task = create_race_task(scenario, scenario_dir=PROJECT_ROOT / "scenarios")
    try:
        policy = create_learner(algorithm, task.spec, learner_params(scenario, task.spec, algorithm))
        capture = Capture(False)
        training = OnPolicyTrainer(task, policy, hooks=[capture], envs_per_worker=2,
                                   scenario=scenario, collection_device="cpu", collection_backend=collection_backend)
        training.train(total_steps=8)
        assert capture.updates and capture.updates[-1]["train/environment_steps"] >= 8
        assert sum(row["train/optimizer_steps"] for row in capture.updates) > 0
        assert capture.updates[-1]["train/learning_rate"] == pytest.approx(1e-4)
    finally:
        task.close()
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("overrides,message", [
    (["algorithm.optimizer=bogus"], "Unsupported optimizer"),
    (["+algorithm.optimizer_kwargs.lr=0.1"], "initial LR"),
    (["+algorithm.optimizer_kwargs.momentum=0.9"], "Unsupported adam optimizer_kwargs"),
    (["+algorithm.optimizer_kwargs.betas=[0.9,1.0]"], "finite"),
    (["+algorithm.optimizer_kwargs.eps=nan"], "finite"),
    (["algorithm.optimizer=sgd", "+algorithm.optimizer_kwargs.nesterov=true"], "nesterov"),
    (["algorithm.params.lr_schedule=cosine", "algorithm.params.learning_rate_end=0.002"], "must not exceed"),
    (["algorithm.params.lr_schedule=cosine", "algorithm.params.learning_rate_end=null"], "requires"),
    (["algorithm.params.lr_schedule=step", "algorithm.params.learning_rate_end=null"], "step_size"),
    (["algorithm.params.lr_schedule=exponential", "algorithm.params.learning_rate_end=null",
      "+algorithm.params.lr_scheduler_kwargs.gamma=2"], "finite"),
    (["algorithm.loss=mappo"], "Unsupported loss"),
    (["algorithm.critic_loss=bogus"], "Unsupported critic_loss"),
])
def test_invalid_configuration_fails_at_application_boundary(overrides, message):
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml", overrides=overrides)
    with pytest.raises(ScenarioError, match=message):
        validate_experiment_scenario(scenario)


def test_hydra_selection_reaches_learner_and_wandb_metadata_and_metric_names(monkeypatch):
    scenario = load_and_expand_scenario("scenarios/ppo_lap_completion_pretrain.yaml", overrides=[
        "algorithm.optimizer=adamw", "+algorithm.optimizer_kwargs.weight_decay=0.02",
        "algorithm.params.lr_schedule=cosine", "algorithm.critic_loss=smooth_l1"])
    validate_experiment_scenario(scenario)
    params = resolve_training_params(scenario["agents"]["car_0"], scenario)
    assert params["optimizer"] == "adamw" and params["optimizer_kwargs"] == {"weight_decay": 0.02}
    assert params["critic_loss"] == "smooth_l1" and params["lr_schedule"] == "cosine"
    calls, metrics = [], []
    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="test", name="test", url="test", config=SimpleNamespace(update=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, define_metric=lambda *a, **k: None, log=lambda data, **kw: metrics.append(data)))
    logger = WandbLogger("test", config=scenario, mode="online")
    config = calls[0]["config"]
    assert config["algorithm/optimizer"] == "adamw"
    assert config["algorithm/optimizer_kwargs/weight_decay"] == 0.02
    assert config["algorithm/params/lr_schedule"] == "cosine"
    assert config["algorithm/critic_loss"] == "smooth_l1"
    policy = agent(**{key: params[key] for key in ("optimizer", "optimizer_kwargs", "critic_loss", "lr_schedule",
                                                 "learning_rate", "learning_rate_end")})
    updated = policy.update(batch(policy))
    updated.update({"train/environment_steps": 4, "train/agent_steps": 4,
                    "train/optimizer_steps_total": updated["train/optimizer_steps"],
                    "perf/end_to_end_env_steps_per_second": 10})
    WandbHook(logger).on_update(updated)
    for key in ("train/learning_rate", "train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl",
                "train/environment_steps", "train/agent_steps", "train/optimizer_steps", "train/optimizer_steps_total",
                "train/action_std", "perf/end_to_end_env_steps_per_second"):
        assert key in metrics[-1]
    logger.log_metrics({"eval/environment_steps": 4, "eval/completion_rate": 0.5,
                        "eval/mean_net_progress": 0.4, "eval/collision_rate": 0.25})
    assert metrics[-1]["eval/completion_rate"] == 0.5
    assert metrics[-1]["eval/mean_net_progress"] == 0.4
    assert metrics[-1]["eval/collision_rate"] == 0.25
