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
from training.on_policy import OnPolicyTrainer as TorchRLMAPPOTrainer
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


def transition(agent, ids, step, reward, terminal, next_state=None, observations=None):
    from tensordict import TensorDict
    observations = observations or {aid: np.full(agent.obs_dim, step, dtype=np.float32) for aid in ids}
    output = agent.sample_batch(ids, observations)
    n = len(IDS)
    agents = TensorDict({
        "observation": torch.zeros(n, agent.obs_dim), "action": torch.zeros(n, agent.action_dim),
        "raw_action": torch.zeros(n, agent.action_dim), "raw_log_prob": torch.zeros(n),
        "index": torch.arange(n), "active": torch.zeros(n, 1, dtype=torch.bool),
    }, [n])
    for aid in ids:
        row = IDS.index(aid)
        agents["observation"][row] = torch.tensor(observations[aid])
        agents["action"][row] = torch.tensor(output.actions[aid])
        agents["raw_action"][row] = torch.tensor(output.raw_actions[aid])
        agents["raw_log_prob"][row] = output.log_probs[aid]
        agents["active"][row] = True
    return TensorDict({"agents": agents, "state": torch.full((agent.global_state_dim,), float(step)),
        "next": TensorDict({
            "state": torch.full((agent.global_state_dim,), float(step + 1 if next_state is None else next_state)),
            "agents": TensorDict({"index": torch.arange(n)}, [n]),
            "team_reward": torch.tensor([float(reward)]),
            "learning": TensorDict({"done": torch.tensor([terminal]),
                                    "terminated": torch.tensor([terminal])}, []),
        }, [])}, [])


def test_native_team_gae_continues_retired_credit_and_filters_actor_padding(monkeypatch):
    agent = learner()
    assert isinstance(agent.loss_module, MAPPOLoss)
    assert isinstance(agent.gae, MultiAgentGAE)
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
    data = torch.stack([transition(agent, ids, step, step + 1, terminal=step == 2)
                        for step, ids in enumerate((IDS, ["car_1"], ["car_1"]))])
    captured = []

    def capture(agent, data, **kwargs):
        captured.append(data.clone())
        return {}

    monkeypatch.setattr("agents.torchrl_mappo.optimize_ppo", capture)
    metrics = agent.update(data)
    data = captured[0]
    assert data.batch_size == (4,)
    assert data["agents", "index"].flatten().tolist() == [0, 1, 1, 1]
    torch.testing.assert_close(data["agents", "value_target"].flatten(), torch.tensor([5.23, 5.23, 4.7, 3.]))
    assert metrics["train/rollout_agent_samples"] == 4


def test_budget_cut_bootstraps_one_shared_global_value(monkeypatch):
    agent = learner()
    with torch.no_grad():
        agent.critic.net[0].weight.fill_(1)
        agent.critic.net[0].bias.zero_()
    data = torch.stack([transition(agent, IDS, 2, 1, terminal=False, next_state=3)])
    captured = []
    monkeypatch.setattr("agents.torchrl_mappo.optimize_ppo", lambda agent, data, **kwargs: captured.append(data.clone()) or {})
    agent.update(data)
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
    data = torch.stack([transition(agent, ["car_1"], 0, 1, terminal=False),
                        transition(agent, ["car_1"], 1, 2, terminal=True)])
    before = {key: value.clone() for key, value in agent.actor.state_dict().items()}
    metrics = agent.update(data)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before[key], value) for key, value in agent.actor.state_dict().items())
    if actor_mode == "independent":
        assert all(torch.equal(before[key], value) for key, value in agent.actor.state_dict().items()
                   if key.startswith("actors.car_0."))
    path = tmp_path / "mappo.pt"
    agent.save(str(path))
    evaluation_policy = MAPPOPolicy(1, 1, *BOUNDS, IDS, params(**extra))
    evaluation_policy.load(str(path))
    payload = torch.load(path, weights_only=False)
    assert payload.pop("network")["architecture"] == "mlp"
    torch.save(payload, path)
    evaluation_policy.load(str(path))
    expected, _ = agent.act_batch(IDS, np.zeros((2, 1), dtype=np.float32), deterministic=True)
    actual, _ = evaluation_policy.act_batch(IDS, np.zeros((2, 1), dtype=np.float32), deterministic=True)
    for aid in IDS:
        np.testing.assert_array_equal(actual[aid], expected[aid])
    evaluation_policy.save(str(path))
    agent.load(str(path))


def test_native_training_preserves_team_rewards_and_fixed_only_continuation():
    class Core(ScriptedEnv):
        possible_agents = (*IDS, "car_2", "car_3")
        trainable_agents = tuple(IDS)
        fixed_policy_agents = ("car_2", "car_3")
        render_mode = None

        def close(self):
            pass

    class Reward(TickReward):
        team_contract = ["tick"]

    core = Core({1: (("car_0",), False, ("car_0",), ()),
                 3: (("car_1",), False, ("car_1",), ()),
                 5: ((), True, (), ("car_2", "car_3"))})
    hooks = Capture(True)
    opponents = {aid: SimpleNamespace(act=lambda obs: np.zeros(2, dtype=np.float32))
                 for aid in core.fixed_policy_agents}
    from tasks import RaceTask
    task = RaceTask(core, policy_agents=IDS, fixed_controllers=opponents,
        obs_composers={aid: ObservationComposer([LidarComponent(1, 10, normalize=False)]) for aid in IDS},
        reward_composers={aid: Reward() for aid in IDS},
        action_composers={aid: ActionComposer([]) for aid in IDS}, team_reward_agent_id=IDS[0])
    training = TorchRLMAPPOTrainer(task, learner(), hooks=[hooks])
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


@pytest.mark.parametrize("actor_mode", ["shared", "independent"])
def test_native_fragments_preserve_masks_and_bootstrap_independently(actor_mode, monkeypatch):
    agent = learner(actor_mode=actor_mode)
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
    fragments = []
    for ids_per_step, rewards in (((IDS, ["car_1"]), (1, 2)), ((IDS,), (100,))):
        fragments.append(torch.stack([transition(agent, ids, step, rewards[step], terminal=False)
                                      for step, ids in enumerate(ids_per_step)]))
    captured = []
    monkeypatch.setattr("agents.torchrl_mappo.optimize_ppo",
                        lambda agent, data, **kwargs: captured.append(data.clone()) or {})
    metrics = agent.update_rollouts(fragments)
    data = captured[0]
    torch.testing.assert_close(data["agents", "value_target"].flatten(), torch.tensor([2.8, 2.8, 2., 100., 100.]))
    assert data["agents", "index"].flatten().tolist() == [0, 1, 1, 0, 1]
    assert metrics["train/rollout_steps"] == 3 and metrics["train/rollout_agent_samples"] == 5
    log_probs = agent.probabilistic_actor.get_dist(data).log_prob(data["agents", "raw_action"])
    torch.testing.assert_close(log_probs, data["agents", "raw_log_prob"])


@pytest.mark.parametrize("actor_mode", ["shared", "independent", "lora"])
def test_legacy_ppo_transfer_extends_inputs_keeps_fresh_critic_and_updates(actor_mode, tmp_path):
    from pathlib import Path
    from agents.common.ppo_policy import PPOPolicy
    from core.scenario import load_and_expand_scenario

    directory = Path(__file__).resolve().parents[1] / "scenarios"
    contracts = []
    for name in ("ppo_lap_completion_pretrain", "mappo_2v2_completion_scratch"):
        scenario = load_and_expand_scenario(str(directory / f"{name}.yaml"))
        composer = ObservationComposer.from_config(scenario["agents"]["car_0"]["observation"],
                                                   scenario["environment"])
        contracts.append(composer.contract)
        assert composer.obs_dim == (158 if len(contracts) == 1 else 192)
    source = PPOPolicy(158, *BOUNDS, {"hidden_dims": [8], "device": "cpu",
                                    "_observation_contract": contracts[0]})
    path = tmp_path / "legacy-ppo.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.pop("network")
    torch.save(checkpoint, path)
    options = params(hidden_dims=[8], vf_hidden_dims=[6],
                     actor_mode="shared" if actor_mode == "lora" else actor_mode,
                     _observation_contract=contracts[1],
                     pretrained_actor_observation_extension="frenet_neighbors")
    if actor_mode == "lora":
        options["lora"] = {"mode": "per_agent", "rank": 2, "per_agent_log_std": True}
    agent = TorchRLMAPPOAgent(192, 3, *BOUNDS, IDS, options)
    fresh_critic = {key: value.clone() for key, value in agent.critic.state_dict().items()}
    agent.load_pretrained_actor(str(path))
    assert all(torch.equal(value, fresh_critic[key]) for key, value in agent.critic.state_dict().items())
    assert not agent.optimizer.state
    observations = torch.randn(2, 192)
    indices = {"adapter_indices": torch.tensor([0, 1])} if agent.routed_actor else {}
    mean, scale = agent.actor(observations, **indices)
    expected_mean, expected_scale = source.actor(observations[:, :158])
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(scale, expected_scale.expand_as(scale))
    before = {key: value.clone() for key, value in agent.actor.state_dict().items()}
    obs = {aid: observations[i].numpy() for i, aid in enumerate(IDS)}
    data = torch.stack([transition(agent, IDS, step, 1, terminal=step == 1, observations=obs)
                        for step in range(2)])
    assert agent.update(data)["train/optimizer_steps"] > 0
    if actor_mode == "lora":
        assert all(not p.requires_grad for p in agent.actor.net.parameters())
        assert all(torch.equal(before[key], value) for key, value in agent.actor.state_dict().items()
                   if key.startswith("net."))
        for i in range(2):
            assert any(not torch.equal(before[key], value) for key, value in agent.actor.state_dict().items()
                       if key.startswith(f"adapters.{i}."))
    elif actor_mode == "independent":
        for aid in IDS:
            assert any(not torch.equal(before[key], value) for key, value in agent.actor.state_dict().items()
                       if key.startswith(f"actors.{aid}."))
        assert agent.actor.actors[IDS[0]].net[0].weight.data_ptr() != agent.actor.actors[IDS[1]].net[0].weight.data_ptr()
    else:
        assert any(not torch.equal(before[key], value) for key, value in agent.actor.state_dict().items())
