from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import IndependentNormal, ProbabilisticActor
from torchrl.objectives.multiagent import MAPPOLoss
from torchrl.objectives.value import MultiAgentGAE

from agents.mappo import MAPPOAgent
from agents.torchrl_updates import optimize_ppo


class _PolicyParameters(torch.nn.Module):
    def __init__(self, actor, routed):
        super().__init__()
        self.actor, self.routed = actor, routed

    def forward(self, observation, index):
        kwargs = {"adapter_indices": index.reshape(-1)} if self.routed else {}
        mean, scale = self.actor(observation.reshape(-1, observation.shape[-1]), **kwargs)
        shape = (*observation.shape[:-1], mean.shape[-1])
        return mean.reshape(shape), scale.expand_as(mean).reshape(shape)


class _CentralValue(torch.nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, state, index):
        # One V(s) for the whole team; actors still receive only local observations.
        return self.critic(state).unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1)


class _DecisionCount:
    def __init__(self):
        self.count = 0

    def size(self):
        return self.count


class TorchRLMAPPOAgent(MAPPOAgent):
    collection_backend = "torchrl"

    def _make_buffers(self):
        self._steps = []
        return {aid: _DecisionCount() for aid in self.agent_ids}

    def __init__(self, obs_dim, global_state_dim, action_low, action_high, agent_ids, params):
        super().__init__(obs_dim, global_state_dim, action_low, action_high, agent_ids, params)
        if (self.critic_mode != "shared_team" or self.reward_mode != "team_shared"
                or self.team_return_mode != "joint"):
            raise ValueError("TorchRL MAPPO requires shared_team critic, team_shared rewards and joint team returns")
        if min(self.n_steps, self.n_epochs, self.batch_size) < 1:
            raise ValueError("MAPPO rollout size, epochs, and batch size must be positive")
        parameters = TensorDictModule(
            _PolicyParameters(self.actor, self.routed_actor),
            in_keys=[("agents", "observation"), ("agents", "index")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        self.probabilistic_actor = ProbabilisticActor(
            parameters, in_keys={"loc": ("agents", "loc"), "scale": ("agents", "scale")},
            out_keys=[("agents", "raw_action")], distribution_class=IndependentNormal,
            return_log_prob=True, log_prob_key=("agents", "raw_log_prob"),
        )
        self.policy = TensorDictSequential(
            self.probabilistic_actor,
            TensorDictModule(torch.tanh, in_keys=[("agents", "raw_action")], out_keys=[("agents", "action")]),
        )
        self.value_module = TensorDictModule(
            _CentralValue(self.critic), in_keys=["state", ("agents", "index")],
            out_keys=[("agents", "state_value")],
        )
        self.loss_module = MAPPOLoss(
            self.probabilistic_actor, self.value_module, clip_epsilon=self.clip_range,
            entropy_bonus=False, critic_coeff=self.vf_coef, loss_critic_type="l2",
            normalize_advantage=False, functional=False,
        )
        self.loss_module.set_keys(
            action=("agents", "raw_action"), sample_log_prob=("agents", "raw_log_prob"),
            value=("agents", "state_value"), advantage=("agents", "advantage"),
            value_target=("agents", "value_target"),
        )
        self.gae = MultiAgentGAE(
            gamma=self.gamma, lmbda=self.gae_lambda, value_network=self.value_module,
            average_gae=False, auto_reset_env=False, time_dim=0,
        )
        self.gae.set_keys(value=("agents", "state_value"), advantage=("agents", "advantage"),
                          value_target=("agents", "value_target"))

    @torch.no_grad()
    def act_batch(self, agent_ids, observations, deterministic=False):
        self._require_lora_source()
        ids = self._validate_agent_batch(agent_ids)
        if not ids:
            return {}, {}
        agents = TensorDict({
            "observation": torch.tensor(self.pack_observations(ids, observations), device=self.device),
            "index": torch.tensor([self._agent_index[aid] for aid in ids], device=self.device),
        }, [len(ids)])
        data = TensorDict({"agents": agents}, [])
        with set_exploration_type(ExplorationType.MODE if deterministic else ExplorationType.RANDOM):
            self.policy(data)
        actions, raw, log_probs = (data["agents", name].cpu().numpy()
                                   for name in ("action", "raw_action", "raw_log_prob"))
        self.last_raw_actions = {aid: raw[i].copy() for i, aid in enumerate(ids)}
        return ({aid: actions[i].copy() for i, aid in enumerate(ids)},
                {aid: float(log_probs[i]) for i, aid in enumerate(ids)})

    def store_batch(self, agent_ids, *, observations, global_state, actions, rewards,
                    log_probs, values, terminated, truncated, raw_actions=None):
        ids = self._validate_agent_batch(agent_ids)
        if not ids:
            return
        if raw_actions is None:
            raise ValueError("TorchRL MAPPO requires stored pre-tanh actions")
        state = torch.tensor(np.asarray(global_state).copy(), dtype=torch.float32, device=self.device)
        if self._steps:
            self._steps[-1]["next", "state"] = state
        n = len(self.agent_ids)
        agents = TensorDict({
            "observation": torch.zeros(n, self.obs_dim, device=self.device),
            "action": torch.zeros(n, self.action_dim, device=self.device),
            "raw_action": torch.zeros(n, self.action_dim, device=self.device),
            "raw_log_prob": torch.zeros(n, device=self.device),
            "index": torch.arange(n, device=self.device),
            "mask": torch.zeros(n, dtype=torch.bool, device=self.device),
        }, [n])
        packed = self.pack_observations(ids, observations)
        for i, aid in enumerate(ids):
            row = self._agent_index[aid]
            for name, value in (("observation", packed[i]), ("action", actions[aid]),
                                ("raw_action", raw_actions[aid]), ("raw_log_prob", log_probs[aid])):
                agents[name][row] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            agents["mask"][row] = True
            self.buffers[aid].count += 1
        self._steps.append(TensorDict({
            "agents": agents, "state": state,
            "next": TensorDict({"state": state.clone(), "agents": TensorDict({"index": agents["index"].clone()}, [n])}, []),
        }, []))

    def store_team_step(self, agent_ids, *, reward, value, terminal):
        if not self._steps:
            raise ValueError("Store agent decisions before their joint team step")
        self._steps[-1]["next"].update({
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "done": torch.tensor([terminal], device=self.device),
            # Joint returns end when no teammate can act, including the finite
            # race horizon. Individual retirement never ends shared team credit.
            "terminated": torch.tensor([terminal], device=self.device),
        })

    def any_buffer_full(self):
        return len(self._steps) >= self.n_steps

    def set_next_state(self, state):
        self._steps[-1]["next", "state"] = torch.tensor(
            np.asarray(state).copy(), dtype=torch.float32, device=self.device,
        )

    @torch.no_grad()
    def actor_actions(self, observations, agent_ids, *, deterministic=False, return_raw=False):
        # Parallel inference may repeat the same actor across independent races.
        self._require_lora_source()
        if len(agent_ids) != len(observations) or any(aid not in self._agent_index for aid in agent_ids):
            raise ValueError("Actor rows require matching, known agent IDs")
        data = TensorDict({"agents": TensorDict({
            "observation": observations,
            "index": torch.tensor([self._agent_index[aid] for aid in agent_ids], device=self.device),
        }, [len(agent_ids)])}, [])
        with set_exploration_type(ExplorationType.MODE if deterministic else ExplorationType.RANDOM):
            self.policy(data)
        result = (data["agents", "action"], data["agents", "raw_log_prob"])
        return (*result, data["agents", "raw_action"]) if return_raw else result

    def clear_buffers(self):
        self._steps.clear()
        for buffer in self.buffers.values():
            buffer.count = 0

    def update(self, next_global_state):
        if not self._steps:
            return {}
        self._require_lora_source()
        self.set_next_state(next_global_state)
        return self.update_rollouts([torch.stack(self._steps)])

    def update_rollouts(self, rollouts):
        self._require_lora_source()
        fragments = []
        for rollout in rollouts:
            fragment = rollout.to(self.device).clone()
            with torch.no_grad():
                self.gae(fragment)
            fragments.append(fragment)
        if not fragments:
            return {}
        data = torch.cat(fragments, dim=0)
        # Keep retired slots in GAE for continuing team credit, then select only
        # actual decisions before advantage normalization and minibatch sampling.
        mask = data["agents", "mask"].reshape(-1)
        states = data["state"].unsqueeze(-2).expand(-1, len(self.agent_ids), -1).reshape(-1, self.global_state_dim)
        selected = data["agents"].reshape(-1)[mask]
        samples = TensorDict({"agents": selected.unsqueeze(-1), "state": states[mask]}, [selected.numel()])
        metrics = optimize_ppo(self, samples, group="agents")
        metrics["train/rollout_steps"] = data.numel()
        metrics["train/rollout_agent_samples"] = selected.numel()
        return metrics
