from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")

from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test, parallel_seed_test

from adapters import RaceGymEnv, RaceParallelEnv
from core.scenario import load_and_expand_scenario
from tasks import RaceTask
from test_race_task import ScriptedEnv, TickReward, make_task
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.ego import LidarComponent


ROOT = Path(__file__).resolve().parents[1]


def scenario(name):
    config = load_and_expand_scenario(str(ROOT / "scenarios" / f"{name}.yaml"))
    config["environment"]["max_steps"] = 6
    return config


@pytest.mark.parametrize("model", ["legacy_st", "combined_slip_st"])
def test_gymnasium_checker_with_real_physics(model, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    config = scenario("ppo_lap_completion_pretrain")
    if model == "legacy_st":
        config["environment"]["vehicle_params"] = {"model": model}
        config["environment"].pop("friction", None)
        agent = config["agents"]["car_0"]
        agent["observation"]["observation"]["frenet_vehicle_track"]["wheel_speed_source"] = "rolling_estimate"
        agent["action_constraints"] = {
            "speed_control": "acceleration", "max_acceleration": 2, "max_deceleration": 2,
        }
    config["environment"]["action_repeat"] = 2
    env = RaceGymEnv.from_scenario(config, scenario_dir=ROOT / "scenarios")
    try:
        assert env.task._snapshot is None  # Construction must not consume a reset.
        assert "render_mode" not in config["environment"]
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=7, options={"map_episode_index": 1, "spawn_episode_index": 2})
        assert env.observation_space.contains(obs)
        assert env.state_space.contains(env.state())
        np.testing.assert_array_equal(info["global_state"], env.state())
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step([0, 0])
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        assert truncated and not terminated
        assert info["race_episode_done"]
        assert info["physics_steps"] == 6
        assert info["decision_physics_steps"] == 2
        with pytest.raises(ResetNeeded):
            env.step([0, 0])
    finally:
        env.close()


def test_pettingzoo_checkers_with_real_two_vs_two_races(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PYGLET_HEADLESS", "true")
    config = scenario("mappo_2v2_completion_scratch")

    def factory():
        return RaceParallelEnv.from_scenario(config, scenario_dir=ROOT / "scenarios")

    env = factory()
    try:
        assert env.possible_agents == ["car_0", "car_1"]
        assert env.task.fixed_policy_agents == ("car_2", "car_3")
        parallel_api_test(env, num_cycles=10)
        assert env.state_space.contains(env.state())
        for aid in env.possible_agents:
            assert env.observation_space(aid) is env.observation_space(aid)
            assert env.action_space(aid) is env.action_space(aid)
    finally:
        env.close()
    parallel_seed_test(factory, num_cycles=10)


def test_parallel_retirement_keeps_terminal_observations_and_shared_rewards_separate():
    core = ScriptedEnv({1: (("learner",), False, ("learner",), ()),
                        2: ((), True, (), ("opponent",))})
    core.render_mode = None
    ids = core.possible_agents
    task = RaceTask(
        core, policy_agents=ids, fixed_controllers={},
        obs_composers={aid: ObservationComposer([LidarComponent(1, 10, normalize=False)]) for aid in ids},
        reward_composers={aid: TickReward() for aid in ids},
        action_composers={aid: ActionComposer([]) for aid in ids},
        action_repeat=3, team_reward_agent_id="learner",
    )
    env = RaceParallelEnv(task)
    with pytest.raises(ResetNeeded):
        env.step({})
    env.reset(seed=7)
    obs, rewards, terms, truncs, infos = env.step({aid: [0, 0] for aid in ids})
    assert all(set(payload) == set(ids) for payload in (obs, rewards, terms, truncs, infos))
    assert env.agents == ["opponent"]
    assert terms == {"learner": True, "opponent": False}
    assert not any(truncs.values())
    assert rewards == {"learner": 1, "opponent": 1}
    assert all(info["team_reward"] == 10 for info in infos.values())
    assert all(info["team_reward_components"] == {"shared": 10} for info in infos.values())
    with pytest.raises(ValueError):
        env.step({aid: [0, 0] for aid in ids})
    assert core.tick == 1
    final_obs, _, final_terms, final_truncs, _ = env.step({"opponent": [0, 0]})
    assert set(final_obs) == {"opponent"}
    assert final_truncs == {"opponent": True} and not any(final_terms.values())
    assert not env.agents
    assert obs["learner"][0] == infos["learner"]["global_state"][0] == 1
    assert env.step({}) == ({}, {}, {}, {}, {})
    assert core.tick == 2


@pytest.mark.parametrize("adapter", [RaceGymEnv, RaceParallelEnv])
def test_policy_episode_can_end_before_the_physical_race(adapter):
    task, controller, _ = make_task({
        1: (("learner",), False, ("learner",), ()),
        2: ((), True, (), ("opponent",)),
    })
    task.env.render_mode = None
    env = adapter(task)
    env.reset(seed=17, options={"episode_index": 2})
    result = env.step([0, 1] if adapter is RaceGymEnv else {"learner": [0, 1]})
    info = result[-1] if adapter is RaceGymEnv else result[-1]["learner"]
    assert not info["race_episode_done"]
    assert info["active_physical_agents"] == ("opponent",)
    assert task.env.reset_args == (17, {"episode_index": 2})
    continuation = env.advance_fixed_agents()
    assert continuation.agent_steps == 0
    assert continuation.physics_steps == 1
    assert continuation.after.episode_done
    assert env.state()[0] == 2
    assert controller.calls == 2
