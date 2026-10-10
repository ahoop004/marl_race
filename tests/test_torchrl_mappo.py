from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("pettingzoo")
pytest.importorskip("torchrl.objectives.multiagent")
from torchrl.objectives.multiagent import MAPPOLoss
from torchrl.objectives.value import MultiAgentGAE

from agents.common.mappo_policy import MAPPOPolicy
from agents.torchrl_mappo import TorchRLMAPPOAgent
from training.torchrl_mappo_trainer import TorchRLMAPPOTrainer
from test_race_task import ScriptedEnv, TickReward
from test_torchrl_ppo import Capture
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.ego import LidarComponent


IDS = ["car_0", "car_1"]
BOUNDS = (np.array([-1, 0], dtype=np.float32), np.array([1, 5], dtype=np.float32))


def params(**extra):
    return {"hidden_dims": [], "device": "cpu", "n_steps": 4,
            "n_epochs": 1, "batch_size": 2, "critic_mode": "shared_team",
            "reward_mode": "team_shared", "team_return_mode": "joint",
            "gamma": 0.9, "gae_lambda": 1, **extra}


def learner(**extra):
    return TorchRLMAPPOAgent(1, 1, *BOUNDS, IDS, params(**extra))


def store(agent, ids, step, reward, terminal):
    observations = {aid: np.array([step], dtype=np.float32) for aid in ids}
    actions, log_probs = agent.act_batch(ids, observations)
    agent.store_batch(
        ids, observations=observations, global_state=np.array([step], dtype=np.float32),
        actions=actions, log_probs=log_probs, rewards=dict.fromkeys(ids, reward),
        values=dict.fromkeys(ids, 0), terminated={aid: aid == "car_0" or terminal for aid in ids},
        truncated=dict.fromkeys(ids, False), raw_actions=agent.last_raw_actions,
    )
    agent.store_team_step(ids, reward=reward, value=0, terminal=terminal)


def test_native_team_gae_continues_retired_credit_and_filters_actor_padding(monkeypatch):
    agent = learner()
    assert isinstance(agent.loss_module, MAPPOLoss)
    assert isinstance(agent.gae, MultiAgentGAE)
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
    for step, ids in enumerate((IDS, ["car_1"], ["car_1"])):
        store(agent, ids, step, step + 1, terminal=step == 2)
    captured = []

    def capture(agent, data, **kwargs):
        captured.append(data.clone())
        return {}

    monkeypatch.setattr("agents.torchrl_mappo.optimize_ppo", capture)
    metrics = agent.update(np.array([1000], dtype=np.float32))
    data = captured[0]
    assert data.batch_size == (4,)
    assert data["agents", "index"].flatten().tolist() == [0, 1, 1, 1]
    torch.testing.assert_close(data["agents", "value_target"].flatten(), torch.tensor([5.23, 5.23, 4.7, 3.]))
    assert metrics["train/rollout_agent_samples"] == 4
    assert agent.buffers["car_0"].size() == 1 and agent.buffers["car_1"].size() == 3


def test_budget_cut_bootstraps_one_shared_global_value(monkeypatch):
    agent = learner()
    with torch.no_grad():
        agent.critic.net[0].weight.fill_(1)
        agent.critic.net[0].bias.zero_()
    store(agent, IDS, 2, 1, terminal=False)
    captured = []
    monkeypatch.setattr("agents.torchrl_mappo.optimize_ppo", lambda agent, data, **kwargs: captured.append(data.clone()) or {})
    agent.update(np.array([3], dtype=np.float32))
    torch.testing.assert_close(captured[0]["agents", "advantage"].flatten(), torch.tensor([1.7, 1.7]))
    torch.testing.assert_close(captured[0]["agents", "value_target"].flatten(), torch.tensor([3.7, 3.7]))


@pytest.mark.parametrize("actor_mode", ["shared", "independent", "lora"])
def test_native_update_routes_actors_and_retains_checkpoint_compatibility(actor_mode, tmp_path):
    extra = {"actor_mode": actor_mode, "hidden_dims": [4]}
    if actor_mode == "lora":
        from agents.common.ppo_policy import PPOPolicy

        extra["actor_mode"] = "shared"
        extra["lora"] = {"mode": "per_agent", "rank": 1, "alpha": 2}
        solo = PPOPolicy(1, *BOUNDS, {"hidden_dims": [4], "device": "cpu"})
        source = tmp_path / "solo.pt"
        solo.save(str(source))
    agent = learner(**extra)
    if actor_mode == "lora":
        agent.load_pretrained_actor(str(source))
    store(agent, ["car_1"], 0, 1, terminal=False)
    store(agent, ["car_1"], 1, 2, terminal=True)
    before = {key: value.clone() for key, value in agent.actor.state_dict().items()}
    metrics = agent.update(np.array([2], dtype=np.float32))
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before[key], value) for key, value in agent.actor.state_dict().items())
    if actor_mode == "independent":
        assert all(torch.equal(before[key], value) for key, value in agent.actor.state_dict().items()
                   if key.startswith("actors.car_0."))
    path = tmp_path / "mappo.pt"
    agent.save(str(path))
    evaluation_policy = MAPPOPolicy(1, 1, *BOUNDS, IDS, params(**extra))
    evaluation_policy.load(str(path))
    expected, _ = agent.act_batch(IDS, np.zeros((2, 1), dtype=np.float32), deterministic=True)
    actual, _ = evaluation_policy.act_batch(IDS, np.zeros((2, 1), dtype=np.float32), deterministic=True)
    for aid in IDS:
        np.testing.assert_array_equal(actual[aid], expected[aid])
    evaluation_policy.save(str(path))
    agent.load(str(path))


def test_pettingzoo_training_preserves_team_rewards_and_fixed_only_continuation():
    class Core(ScriptedEnv):
        possible_agents = (*IDS, "car_2", "car_3")
        trainable_agents = tuple(IDS)
        fixed_policy_agents = ("car_2", "car_3")
        render_mode = None

    class Reward(TickReward):
        team_contract = ["tick"]

    core = Core({1: (("car_0",), False, ("car_0",), ()),
                 3: (("car_1",), False, ("car_1",), ()),
                 5: ((), True, (), ("car_2", "car_3"))})
    hooks = Capture(True)
    opponents = {aid: SimpleNamespace(act=lambda obs: np.zeros(2, dtype=np.float32))
                 for aid in core.fixed_policy_agents}
    training = TorchRLMAPPOTrainer(
        core, learner(), IDS, opponents,
        {aid: ObservationComposer([LidarComponent(1, 10, normalize=False)]) for aid in IDS},
        {aid: Reward() for aid in IDS}, ActionComposer([]), hooks=[hooks], reward_mode="team_shared",
    )
    training.train(total_steps=6)
    assert training._environment_steps == training._physics_steps == training._agent_steps == 6
    assert len(hooks.steps) == 6
    assert [record.step_idx for record in hooks.steps] == [0, 0, 1, 2, 0, 0]
    assert [record.reward for record in hooks.steps] == [11, 11, 11, 11.5, 11, 11]
    assert len(hooks.episodes) == 1 and hooks.episodes[0][1] == 33.5
    assert hooks.episodes[0][3]["episode_steps"] == 5
    assert [row["train/rollout_agent_samples"] for row in hooks.updates] == [4, 2]
    assert [row["train/environment_steps"] for row in hooks.updates] == [5, 6]
    assert hooks.ended


def test_parallel_native_fragments_preserve_masks_and_do_not_mix_races(monkeypatch):
    from multiprocessing.reduction import ForkingPickler
    import pickle
    from training.torchrl_collectors import MAPPOCollectorAgent, deserialize_rollout, serialize_rollout

    agent = learner()
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
    contract = {name: getattr(agent, name) for name in (
        "agent_ids", "obs_dim", "obs_dims", "global_state_dim", "action_dim",
    )}
    fragments = []
    for ids_per_step, rewards in (((IDS, ["car_1"]), (1, 2)), ((IDS,), (100,))):
        collector = MAPPOCollectorAgent(contract, 2)
        for step, ids in enumerate(ids_per_step):
            observations = {aid: np.array([step], dtype=np.float32) for aid in ids}
            actions, log_probs = agent.act_batch(ids, observations)
            collector.store_batch(ids, observations=observations, global_state=np.array([step]),
                                  actions=actions, log_probs=log_probs, raw_actions=agent.last_raw_actions,
                                  rewards={}, values={}, terminated={}, truncated={})
            collector.store_team_step(ids, reward=rewards[step], value=0, terminal=False)
            collector.set_next_state(np.array([step + 1]))
        collector.finish_fragment({})
        assert not collector._steps and all(buf.size() == 0 for buf in collector.buffers.values())
        fragments.extend(deserialize_rollout(pickle.loads(ForkingPickler.dumps(serialize_rollout(fragment))))
                         for fragment in collector.take_fragments())
    captured = []
    monkeypatch.setattr("agents.torchrl_mappo.optimize_ppo",
                        lambda agent, data, **kwargs: captured.append(data.clone()) or {})
    metrics = agent.update_rollouts(fragments)
    torch.testing.assert_close(captured[0]["agents", "value_target"].flatten(), torch.tensor([2.8, 2.8, 2., 100., 100.]))
    assert captured[0]["agents", "index"].flatten().tolist() == [0, 1, 1, 0, 1]
    assert metrics["train/rollout_steps"] == 3 and metrics["train/rollout_agent_samples"] == 5


@pytest.mark.parametrize("actor_mode", ["shared", "independent"])
def test_parallel_inference_repeated_ids_match_native_log_probabilities(actor_mode):
    from tensordict import TensorDict
    from training.parallel_mappo import infer_requests

    agent = learner(actor_mode=actor_mode)
    observations = np.array([[1], [2], [3]], dtype=np.float32)
    requests = {0: ("act", (IDS, observations[:2], np.array([0], dtype=np.float32))),
                1: ("act", (["car_1"], observations[2:], np.array([100], dtype=np.float32)))}
    responses = infer_requests(agent, requests)
    rows = [(0, "car_0"), (0, "car_1"), (1, "car_1")]
    data = TensorDict({"agents": TensorDict({
        "observation": torch.tensor(observations), "index": torch.tensor([0, 1, 1]),
    }, [3])}, [])
    raw = torch.tensor(np.stack([responses[key][3][aid] for key, aid in rows]))
    expected = agent.probabilistic_actor.get_dist(data).log_prob(raw)
    actual = torch.tensor([responses[key][1][aid] for key, aid in rows])
    torch.testing.assert_close(actual, expected)
