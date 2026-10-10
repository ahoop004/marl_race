from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torchrl")
from tensordict import TensorDict, TensorDictBase
from torchrl.collectors import Collector
from torchrl.envs import step_mdp
from torchrl.envs.utils import check_env_specs
from torchrl.objectives.value import GAE, MultiAgentGAE

from adapters import RaceGymEnv, RaceParallelEnv
from adapters.native_torchrl import NativeRaceTorchRLEnv
from core.scenario import load_and_expand_scenario
from tasks import RaceTask
from test_race_task import ScriptedEnv, TickReward, make_task
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.ego import LidarComponent


ROOT = Path(__file__).resolve().parents[1]
IDS = ("car_0", "car_1")


def team_task(events, repeat=1):
    class Core(ScriptedEnv):
        possible_agents = (*IDS, "car_2", "car_3")
        trainable_agents = IDS
        fixed_policy_agents = ("car_2", "car_3")
        render_mode = None

        def close(self):
            self.closed = True

    core = Core(events)
    controllers = {aid: SimpleNamespace(act=lambda obs: np.zeros(2, dtype=np.float32))
                   for aid in core.fixed_policy_agents}
    return RaceTask(
        core, policy_agents=IDS, fixed_controllers=controllers,
        obs_composers={aid: ObservationComposer([LidarComponent(1, 10, normalize=False)]) for aid in IDS},
        reward_composers={aid: TickReward() for aid in IDS},
        action_composers={aid: ActionComposer([]) for aid in IDS},
        action_repeat=repeat, team_reward_agent_id=IDS[0],
    )


def zero_policy(data: TensorDictBase):
    data["agents", "action"] = torch.zeros_like(data["agents", "observation"][..., :1]).expand(
        *data["agents", "observation"].shape[:-1], 2).clone()
    return data


def single_task(events=None, repeat=3):
    task, controller, _ = make_task(events or {}, repeat=repeat)
    task.env.render_mode = None
    task.env.close = lambda: None
    return task, controller


def test_construction_specs_seed_options_and_snapshot_ownership():
    task, controller = single_task()
    env = NativeRaceTorchRLEnv(task)
    try:
        assert task._snapshot is None and controller.resets == 0
        assert env.batch_size == () and env.group_map == {"agents": ["learner"]}
        assert env.action_spec["agents", "action"].shape == (1, 2)
        assert env.observation_spec["agents", "observation"].shape == (1, 1)
        assert env.observation_spec["agents", "done"].shape == (1, 1)
        assert env.observation_spec["physical", "active"].shape == (2, 1)
        assert set(env.done_keys) == {"done", "terminated", "truncated"}
        rng = torch.random.get_rng_state().clone()
        env.set_seed(17)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        assert task._snapshot is None
        current = env.reset(options={"episode_index": 2})
        initial = current.clone()
        assert task.env.reset_args == (17, {"episode_index": 2})
        assert env.observation_spec.is_in(current)
        current["agents", "action"] = torch.tensor([[0.0, 1.0]])
        next_data = env.step(current)["next"]
        final = next_data.clone()
        assert next_data["agents", "reward"].item() == 6
        assert next_data["physics_steps"].item() == 3
        assert next_data["decision_steps"].item() == 1
        assert next_data["agent_steps"].item() == 1
        assert next_data["elapsed_seconds"].item() == pytest.approx(0.15)
        assert all(np.allclose(row["learner"], [0, 0.3]) for row in task.env.calls)
        env.reset(seed=23, options={"episode_index": 4})
        torch.testing.assert_close(initial["agents", "observation"], torch.zeros(1, 1))
        torch.testing.assert_close(final["agents", "observation"], torch.full((1, 1), 3.0))
        torch.testing.assert_close(next_data["state"], final["state"])
        torch.testing.assert_close(next_data["agents", "observation"], final["agents", "observation"])
        assert task.env.reset_args == (23, {"episode_index": 4})
        env.reset()
        assert task.env.reset_args == (None, None)
    finally:
        env.close()
        env.close()


@pytest.mark.parametrize("mode,reduction", [("individual", "mean"), ("team_shared", "mean"), ("team_shared", "sum")])
@pytest.mark.parametrize("repeat", [1, 3])
def test_scripted_parallel_parity_retirement_and_fixed_only_continuation(mode, reduction, repeat):
    events = {1: (("car_0",), False, ("car_0",), ()),
              3: (("car_1",), False, ("car_1",), ()),
              5: ((), True, (), ("car_2", "car_3"))}
    native = NativeRaceTorchRLEnv(team_task(events, repeat), reward_mode=mode,
                                 team_reward_reduction=reduction)
    parallel = RaceParallelEnv(team_task(events, repeat), reward_mode=mode,
                               team_reward_reduction=reduction)
    observed = []
    native.on_physics_step = observed.append
    try:
        current = native.reset(seed=31)
        parallel.reset(seed=31)
        saved_terminal = None
        while not native.snapshot.episode_done:
            actors = tuple(native.task.agents)
            action = {aid: np.array([0.1, 0.2], dtype=np.float32) for aid in actors}
            current["agents", "action"] = torch.tensor([[0.1, 0.2], [0.1, 0.2]])
            transition = native.step(current)
            nxt = transition["next"]
            if actors:
                obs, rewards, terms, truncs, _ = parallel.step(action)
                for i, aid in enumerate(IDS):
                    if aid in actors:
                        np.testing.assert_array_equal(nxt["agents", "observation"][i].numpy(), obs[aid])
                        assert nxt["agents", "reward"][i].item() == rewards[aid]
                        assert nxt["agents", "terminated"][i].item() == terms[aid]
                        assert nxt["agents", "truncated"][i].item() == truncs[aid]
                    else:
                        assert nxt["agents", "reward"][i].item() == 0
                        assert nxt["agents", "individual_reward"][i].item() == 0
            else:
                parallel.advance_fixed_agents()
                assert not nxt["learning", "valid"].item()
                assert nxt["team_reward"].item() == 0
                assert not transition["agents", "active"].any()
            np.testing.assert_array_equal(nxt["state"].numpy(), parallel.state())
            assert nxt["done"].item() == parallel.snapshot.episode_done
            assert nxt["physics_steps"].item() == parallel.snapshot.physics_steps
            assert nxt["shared_bonus"].item() == parallel.last_step.team_reward
            assert nxt["learning", "done"].item() == (not native.task.agents)
            assert nxt["agents"].batch_size == (2,)
            if saved_terminal is None:
                saved_terminal = nxt["agents", "observation"][0].clone()
                assert transition["agents", "active"][0].item()
                assert not nxt["agents", "active"][0].item()
                assert not nxt["done"].item() and not nxt["learning", "done"].item()
            torch.testing.assert_close(nxt["agents", "observation"][0], saved_terminal)
            current = step_mdp(transition)
        assert len(observed) == 5
        assert nxt["truncated"].item() and not nxt["terminated"].item()
        assert nxt["learning", "terminated"].item()
        assert not nxt["learning", "truncated"].item()
        assert not nxt["physical", "active"].any()
        for left, right in zip(native.task.env.calls, parallel.task.env.calls, strict=True):
            assert left.keys() == right.keys()
            for aid in left:
                np.testing.assert_array_equal(left[aid], right[aid])
        with pytest.raises(RuntimeError):
            native.step(zero_policy(current))
    finally:
        native.close()
        parallel.close()


@pytest.mark.parametrize("termination", [True, False])
def test_physical_boundary_uses_final_active_opponent_not_retired_learner(termination):
    # An earlier learner truncation cannot turn a later natural finish into a
    # root truncation; a final opponent timeout cannot become a termination.
    events = {1: (("learner",), False, (), ("learner",)),
              2: ((), True, ("opponent",) if termination else (),
                  () if termination else ("opponent",))}
    task, controller = single_task(events)
    env = NativeRaceTorchRLEnv(task)
    try:
        current = env.reset(seed=9)
        first, current = env.step_and_maybe_reset(zero_policy(current))
        assert first["next", "agents", "truncated"].item()
        assert first["next", "learning", "done"].item()
        assert not first["next", "done"].item()
        assert controller.resets == 1
        second, current = env.step_and_maybe_reset(zero_policy(current))
        assert second["next", "terminated"].item() == termination
        assert second["next", "truncated"].item() != termination
        assert second["next", "agent_steps"].item() == 0
        assert controller.resets == 2
        assert current["agents", "active"].item()
    finally:
        env.close()


@pytest.mark.parametrize("truncations", [(), ("learner",)])
def test_task_closure_final_observation_and_truncation(truncations):
    task, _ = single_task({1: ((), True, (), truncations)})
    env = NativeRaceTorchRLEnv(task)
    try:
        transition = env.step(zero_policy(env.reset()))
        nxt = transition["next"]
        assert nxt["done"].item() and nxt["truncated"].item()
        assert nxt["agents", "truncated"].item()
        assert nxt["agents", "observation"].item() == 1
        assert nxt["decision_physics_steps"].item() == 1
    finally:
        env.close()


@pytest.mark.parametrize("boundary,expected", [("terminated", 1.0), ("truncated", 3.7), ("cut", 3.7)])
def test_individual_gae_uses_final_observation_and_individual_boundary(boundary, expected):
    events = {} if boundary == "cut" else {
        1: (("learner",), False,
            ("learner",) if boundary == "terminated" else (),
            ("learner",) if boundary == "truncated" else ())}
    task, _ = single_task(events, repeat=1)
    env = NativeRaceTorchRLEnv(task)
    try:
        data = env.step(zero_policy(env.reset())).unsqueeze(0)
        assert not data["next", "done"].item()
        assert data["next", "agents", "observation"].item() == 1
        data["agents", "state_value"] = torch.full((1, 1, 1), 2.0)
        data["next", "agents", "state_value"] = torch.full((1, 1, 1), 3.0)
        gae = GAE(gamma=0.9, lmbda=1, value_network=None, time_dim=0)
        gae.set_keys(value=("agents", "state_value"), reward=("agents", "reward"),
                     done=("agents", "done"), terminated=("agents", "terminated"),
                     advantage=("agents", "advantage"), value_target=("agents", "value_target"))
        gae(data)
        assert data["agents", "value_target"].item() == pytest.approx(expected)
    finally:
        env.close()


def test_invalid_active_actions_do_not_advance_task_and_inactive_slots_are_ignored():
    task, controller = single_task({1: (("learner",), False, ("learner",), ())}, repeat=1)
    env = NativeRaceTorchRLEnv(task)
    try:
        with pytest.raises(RuntimeError):
            env.step(TensorDict({"agents": {"action": torch.zeros(1, 2)}}, []))
        current = env.reset()
        for action in (torch.zeros(1, 3), torch.zeros(1, 2, dtype=torch.float64),
                       torch.tensor([[0.0, float("nan")]]), torch.tensor([[0.0, 2.0]])):
            current["agents", "action"] = action
            with pytest.raises(ValueError):
                env.step(current)
            assert task.env.tick == 0 and controller.calls == 0
        current = step_mdp(env.step(zero_policy(current)))
        current["agents", "action"] = torch.full((1, 2), float("nan"))
        continuation = env.step(current)["next"]
        assert continuation["agent_steps"].item() == 0
        assert task.env.tick == 2 and controller.calls == 2
    finally:
        env.close()


def test_direct_collector_keeps_retired_slots_and_waits_for_physical_completion():
    events = {1: (("car_0",), False, ("car_0",), ()),
              3: (("car_1",), False, ("car_1",), ()),
              5: ((), True, (), ("car_2", "car_3"))}
    env = NativeRaceTorchRLEnv(team_task(events), reward_mode="team_shared")
    collector = Collector(env, zero_policy, backend="direct", frames_per_batch=6,
                          total_frames=6, reset_at_each_iter=False, set_truncated=False)
    try:
        data = next(iter(collector))
        assert data.batch_size == (6,) and data["agents"].batch_size == (6, 2)
        assert data["agents", "active"].sum().item() == 6
        assert data["next", "done"].flatten().tolist() == [False, False, False, False, True, False]
        assert data["next", "learning", "done"].flatten().tolist() == [False, False, True, True, True, False]
        assert data["next", "learning", "valid"].flatten().tolist() == [True, True, True, False, False, True]
        assert data["collector", "traj_ids"][:5].unique().numel() == 1
        assert data["collector", "traj_ids"][5] != data["collector", "traj_ids"][0]
        # Read team learning flags explicitly; physical continuation contributes
        # no rewards after the learner endpoint. Earlier retired actors retain credit.
        data["agents", "state_value"] = torch.zeros(6, 2, 1)
        data["next", "agents", "state_value"] = torch.zeros(6, 2, 1)
        gae = MultiAgentGAE(gamma=0.9, lmbda=1, value_network=None, time_dim=0)
        gae.set_keys(value=("agents", "state_value"), reward="team_reward",
                     done=("learning", "done"), terminated=("learning", "terminated"),
                     advantage=("agents", "advantage"), value_target=("agents", "value_target"))
        gae(data)
        torch.testing.assert_close(data["agents", "value_target"][0, :, 0], torch.full((2,), 30.215))
        assert data["agents", "value_target"][3:5].eq(0).all()
    finally:
        collector.shutdown()


@pytest.mark.parametrize("name,repeat", [("ppo_lap_completion_pretrain", 1),
                                         ("ppo_lap_completion_pretrain", 2),
                                         ("mappo_2v2_completion_scratch", 1)])
def test_real_physics_seeded_adapter_parity_specs_and_short_rollouts(name, repeat, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    config = load_and_expand_scenario(str(ROOT / "scenarios" / f"{name}.yaml"))
    config["environment"].update(max_steps=6, action_repeat=repeat)
    config["environment"].update(map_bundles_train=["circle_map", "Budapest_map"],
                                 map_cycle="per_episode", map_pick="round_robin")
    team = name.startswith("mappo")
    mode = "team_shared" if team else "individual"
    native = NativeRaceTorchRLEnv.from_scenario(config, scenario_dir=ROOT / "scenarios", reward_mode=mode)
    adapter = (RaceParallelEnv if team else RaceGymEnv).from_scenario(
        config, scenario_dir=ROOT / "scenarios", **({"reward_mode": mode} if team else {}))
    try:
        assert native.task._snapshot is None and adapter.task._snapshot is None
        for episode_index, seed in enumerate((7, 19), start=2):
            options = {"map_episode_index": episode_index, "spawn_episode_index": 3}
            current = native.reset(seed=seed, options=options)
            initial = current.clone()
            initial_metadata = native.task.episode_metadata
            current = native.reset(seed=seed, options=options)
            for key in initial.keys(include_nested=True, leaves_only=True):
                torch.testing.assert_close(current[key], initial[key], rtol=0, atol=0)
            adapter.reset(seed=seed, options=options)
            assert native.task.episode_metadata.map_id == (
                "circle_map" if episode_index == 2 else "Budapest_map")
            assert native.task.episode_metadata.map_id == adapter.task.episode_metadata.map_id
            assert native.task.episode_metadata.environment_seed == seed
            for aid in native.physical_agent_ids:
                for field in ("pose", "velocity"):
                    expected = initial_metadata.spawn_configuration["initial_states"][aid][field]
                    np.testing.assert_array_equal(
                        native.task.episode_metadata.spawn_configuration["initial_states"][aid][field], expected)
                    np.testing.assert_array_equal(
                        adapter.task.episode_metadata.spawn_configuration["initial_states"][aid][field], expected)
            np.testing.assert_array_equal(current["state"].numpy(), adapter.state())
            for i, aid in enumerate(native.agent_ids):
                np.testing.assert_array_equal(current["agents", "observation"][i].numpy(), adapter.snapshot.observations[aid])
            while not native.snapshot.episode_done:
                actors = tuple(native.task.agents)
                actions = {aid: np.array([0.05, 0.1], dtype=np.float32) for aid in actors}
                current["agents", "action"] = torch.tensor([[0.05, 0.1]] * len(native.agent_ids))
                transition = native.step(current)
                nxt = transition["next"]
                if team:
                    if actors:
                        adapter.step(actions)
                    else:
                        adapter.advance_fixed_agents()
                else:
                    adapter.step(actions[native.agent_ids[0]])
                np.testing.assert_array_equal(nxt["state"].numpy(), adapter.state())
                for i, aid in enumerate(native.agent_ids):
                    decision = adapter.last_step.decisions.get(aid)
                    if decision is not None:
                        np.testing.assert_array_equal(nxt["agents", "observation"][i].numpy(), decision.next_observation)
                        np.testing.assert_array_equal(native.last_step.decisions[aid].action_physical, decision.action_physical)
                        assert nxt["agents", "terminated"][i].item() == decision.terminated
                        assert nxt["agents", "truncated"][i].item() == decision.truncated
                        learning_reward = adapter.last_rewards[aid] if team else decision.individual_reward
                        assert nxt["agents", "reward"][i].item() == pytest.approx(learning_reward, abs=1e-7)
                assert nxt["done"].item() == adapter.snapshot.episode_done
                assert torch.isfinite(nxt["agents", "observation"]).all()
                for key in nxt["physical"].keys():
                    np.testing.assert_array_equal(
                        nxt["physical", key].flatten().numpy(),
                        adapter.snapshot.global_state.masks[f"{key}_mask"])
                assert native.last_step.physics_steps == adapter.last_step.physics_steps
                assert native.last_step.team_reward == adapter.last_step.team_reward
                current = step_mdp(transition)
        check_env_specs(native, seed=17)
        native.set_seed(17)
        rollout = native.rollout(6, zero_policy, break_when_any_done=True)
        assert rollout["agents"].batch_size == (*rollout.batch_size, len(native.agent_ids))
        assert rollout["next", "done"][-1].item()
        assert rollout["next", "physics_steps"][-1].item() <= 6
        assert native.full_action_spec.is_in(rollout[0].select("agents"))
    finally:
        native.close()
        adapter.close()
