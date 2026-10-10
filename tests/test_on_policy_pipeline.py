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


@pytest.mark.parametrize("measure_mpc", [False, True])
def test_stage_profiling_preserves_updates_and_restores_instance_methods(measure_mpc):
    plain_capture, measured_capture = Capture(False), Capture(False)
    plain, _ = trainer(plain_capture)
    measured, _ = trainer(measured_capture)
    measured.measure_stages = True
    measured.measure_mpc = measure_mpc
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


@pytest.fixture
def shooting_controller(monkeypatch):
    """Exercise MPC control flow with a cheap shooting function and counted solver."""
    import numpy as np
    from agents.mpc import racing

    traffic = np.zeros((1, 5))
    clearance = [1.0]
    shoots, solves = [], []

    def shoot(controls, *args, record_trajectory=True):
        shoots.append(record_trajectory)
        return float(np.sum(controls**2)), clearance[0], np.zeros((2, 8))

    shoot.signatures = []

    def minimize(objective, flat, **kwargs):
        solves.append(kwargs)
        for _ in range(3):
            objective(flat)
        return SimpleNamespace(x=flat.copy(), nfev=3, nit=2, success=False, status=1)

    monkeypatch.setattr(racing, "_shoot", shoot)
    monkeypatch.setattr(racing, "minimize", minimize)

    def make(aid="car_2"):
        controller = racing.RacingMPCAgent(dict(agent_id=aid))
        controller.env = SimpleNamespace(
            params={"mu": 1.0},
            get_agent_state=lambda _: SimpleNamespace(velocity=[1., 0.], angular_velocity=0.),
            get_global_state=lambda: SimpleNamespace(metadata={}))
        controller.p = np.ones(13)
        controller.dt, controller.radius = .05, .1
        controller.steer_min, controller.steer_max = -.4, .4
        controller.path = np.column_stack((np.arange(150)*.1, np.zeros(150)))
        controller.field = np.ones((2, 2))
        controller.origin, controller.resolution = np.zeros(3), .1
        controller.footprint = np.zeros((1, 2))
        controller._map = lambda: None
        controller._traffic = lambda *args: traffic
        return controller

    obs = dict(pose=np.zeros(3), velocity=np.array([1., 0.]), steering_angle=0.,
               wheel_speed=10., wheel_speed_reference=10., steering_reference=0.)
    return SimpleNamespace(make=make, obs=obs, shoots=shoots, solves=solves, clearance=clearance)


@pytest.mark.parametrize("fallback", [False, True])
def test_mpc_profile_counts_solver_calls_and_preserves_actions(shooting_controller, monkeypatch, fallback):
    import numpy as np
    from agents.mpc import racing

    fixture = shooting_controller
    fixture.clearance[0] = -1. if fallback else 1.
    plain, measured = fixture.make(), fixture.make()
    records = []
    measured.profile_mpc = records.append
    clock = racing.perf_counter
    # Disabled instrumentation must not read the clock, including inside objective.
    monkeypatch.setattr(racing, "perf_counter", lambda: pytest.fail("disabled MPC read the clock"))
    expected = plain.act(fixture.obs)
    monkeypatch.setattr(racing, "perf_counter", clock)
    fixture.shoots.clear()
    fixture.solves.clear()
    actual = measured.act(fixture.obs)
    np.testing.assert_array_equal(actual, expected)
    row = records[-1]
    assert row["optimization_starts"] == len(fixture.solves) == 3
    assert row["nfev"] == fixture.shoots.count(False)-1 == 9
    assert row["nit"] == 6 and row["solver_failures"] == 3
    assert row["solver_statuses"] == {"1": 3}
    assert row["outside_trajectory_evaluations"] == fixture.shoots.count(True)+1 == 10
    assert row["alternative_starts"] == 2 and row["alternative_start_decisions"] == 1
    assert row["braking_fallback"] == fallback
    assert row["braking_selected"] == 1
    assert not row["warm_available"] and not row["warm_used"] and row["new_plan"]
    assert row["cold_start"] == 1
    assert row["solver_seconds"] >= row["objective_seconds"] > 0
    assert sum(row["stages"].values()) == pytest.approx(row["seconds"], abs=1e-12)
    assert all(math.isfinite(value) and value >= 0 for value in row["stages"].values())
    measured.act(fixture.obs)
    assert records[-1]["warm_used"] == (not fallback)
    assert not records[-1]["cold_start"]
    assert records[-1]["optimization_starts"] == (3 if fallback else 1)
    measured.reset()
    assert measured.profile_mpc is not None
    measured.act(fixture.obs)
    assert len(records) == 3 and records[-1]["cold_start"]
    assert not records[-1]["warm_used"]


def test_mpc_profile_adapter_aggregation_tracks_live_retirement(shooting_controller):
    from contextlib import ExitStack
    from env.collision_state import RaceLifecycle
    from wrappers.actions.composer import WheelReferenceAdapter

    fixture = shooting_controller
    training, _ = trainer(Capture(False))
    training.trainable_ids = ["car_0", "car_1"]
    training.measure_mpc = training.measure_stages = True
    controllers = {aid: fixture.make(aid) for aid in ("car_2", "car_3")}
    training.task.fixed_controllers = {aid: WheelReferenceAdapter(controller, dict(
        wheel_radius=.1, steering_min=-.4, steering_max=.4, wheel_speed_min=0., wheel_speed_max=100.))
        for aid, controller in controllers.items()}
    lifecycle = RaceLifecycle(("car_0", "car_1", "car_2", "car_3"), target_laps=1)
    with ExitStack() as resources:
        training._instrument_mpc(resources)
        training._timing_phase = "warmup"
        for removed in (None, "car_0", "car_1", "car_2"):
            if removed:
                if removed == "car_1":
                    lifecycle.record_lap_crossing(removed, step=0)
                else:
                    lifecycle.record_collision(removed, step=0)
            training.task.env.agents = list(lifecycle.active_agents)
            for aid, adapter in training.task.fixed_controllers.items():
                if aid in training.task.env.agents:
                    training.measure_stage("fixed_controller", adapter.act, fixture.obs)
            training.mpc_environment_decisions["warmup"] = training.mpc_environment_decisions.get("warmup", 0)+1
        summary = training.mpc_summary()
        assert summary["overall"]["calls"] == training.stage_calls["warmup"]["fixed_controller"] == 7
        assert summary["agent"]["car_2"]["calls"] == 3
        assert summary["agent"]["car_3"]["calls"] == 4
        assert {key: row["calls"] for key, row in summary["active_vehicles"].items()} == {
            "4": 2, "3": 2, "2": 2, "1": 1}
        assert summary["active_learners"]["0"]["calls"] == 3
        assert summary["active_mpc"]["1"]["calls"] == 1
        assert summary["traffic"]["1"]["calls"] == 7
        assert summary["overall"]["environment_decisions"] == 4
        assert summary["overall"]["mean_nfev_per_optimization"] == 3
        assert summary["overall"]["solver_failure_rate"] == 1
        assert summary["phase"]["initialization"]["calls"] == 0
        assert summary["overall"]["seconds"] <= training.stage_seconds["warmup"]["fixed_controller"]
        json.dumps(summary, allow_nan=False)
    assert all(controller.profile_mpc is None for controller in controllers.values())
    assert len(fixture.solves) == summary["overall"]["optimization_starts"]


@pytest.mark.parametrize("warmup_updates", [0, 1])
def test_mpc_profile_phase_boundary_and_run_reset_without_stage_timing(tmp_path, warmup_updates):
    training, _ = trainer(Capture(False))
    training.measure_mpc = True
    training.profiling_warmup_updates = warmup_updates
    training.hooks.append(CSVHook(CSVLogger(tmp_path)))
    training.train(total_steps=9)
    assert training.mpc_environment_decisions == ({"warmup": 4, "steady": 5} if warmup_updates else {"steady": 9})
    assert training.mpc_summary()["overall"]["calls"] == 0  # No racing MPC in this fixture.
    assert json.loads((tmp_path / "run_summary.json").read_text())["mpc_profile"]["overall"]["environment_decisions"] == 9
    training.hooks.clear()
    training.train(total_steps=12)
    assert sum(training.mpc_environment_decisions.values()) == 3
    assert training.mpc_totals == {}


def test_mpc_map_change_invalidates_warm_plan_without_resetting_run_measurement(shooting_controller):
    fixture = shooting_controller
    controller = fixture.make()
    records = []
    controller.profile_mpc = records.append
    controller.act(fixture.obs)
    assert controller._warm is not None
    controller._map = lambda: setattr(controller, "_warm", None)
    controller.act(fixture.obs)
    assert records[-1]["warm_available"] == 1
    assert records[-1]["warm_used"] == 0 and records[-1]["new_plan"] == 1
    assert len(records) == 2


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_mpc_instrumentation_preserves_policy_optimizer_results(shooting_controller, algorithm):
    from test_native_torchrl_env import team_task
    from test_torchrl_mappo import params, BOUNDS
    from agents.torchrl_mappo import TorchRLMAPPOAgent
    from training.on_policy import OnPolicyTrainer
    from wrappers.actions.composer import WheelReferenceAdapter

    fixture = shooting_controller

    def make():
        if algorithm == "ppo":
            training, _ = trainer(Capture(False))
        else:
            task = team_task({3: (("car_0", "car_1", "car_2", "car_3"), True,
                                      ("car_0", "car_1", "car_2", "car_3"), ())})
            policy = TorchRLMAPPOAgent(1, task.state_space().shape[0], *BOUNDS, IDS, params(n_steps=4))
            training = OnPolicyTrainer(task, policy, hooks=[Capture(False)])
        for aid in training.task.fixed_controllers:
            controller = fixture.make(aid)
            act = controller.act
            controller.act = lambda obs, act=act: act(fixture.obs)
            training.task.fixed_controllers[aid] = WheelReferenceAdapter(controller, dict(
                wheel_radius=.1, steering_min=-.4, steering_max=.4, wheel_speed_min=0., wheel_speed_max=100.))
        return training

    plain, measured = make(), make()
    measured.measure_mpc = True
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
    summary = measured.mpc_summary()
    assert summary["overall"]["calls"] == 9*(1 if algorithm == "ppo" else 2)
    assert sum(summary["phase"][phase]["calls"] for phase in summary["phase"]) == summary["overall"]["calls"]
    assert all(adapter.controller.profile_mpc is None for adapter in measured.task.fixed_controllers.values())
