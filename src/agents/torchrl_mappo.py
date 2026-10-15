from __future__ import annotations

from contextlib import nullcontext

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import IndependentNormal, ProbabilisticActor
from torchrl.objectives.value import MultiAgentGAE

from agents.common.mappo_policy import MAPPOPolicy
from agents.common.optimization import make_objective
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


class TorchRLMAPPOAgent(MAPPOPolicy):
    def __init__(self, obs_dim, global_state_dim, action_low, action_high, agent_ids, params):
        super().__init__(obs_dim, global_state_dim, action_low, action_high, agent_ids, params, training=True)
        self.frames_per_batch = int(params.get("frames_per_batch", 256))
        if (self.critic_mode != "shared_team" or self.reward_mode != "team_shared"
                or self.team_return_mode != "joint"):
            raise ValueError("TorchRL MAPPO requires shared_team critic, team_shared rewards and joint team returns")
        if min(self.n_steps, self.n_epochs, self.batch_size, self.frames_per_batch) < 1:
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
        self.collection_policy = self.policy
        self.value_module = TensorDictModule(
            _CentralValue(self.critic), in_keys=["state", ("agents", "index")],
            out_keys=[("agents", "state_value")],
        )
        self.loss_module = make_objective(
            self.algorithm, self.loss, self.critic_loss, self.probabilistic_actor, self.value_module,
            clip_epsilon=self.clip_range, entropy_bonus=False, critic_coeff=self.vf_coef,
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
            deactivate_vmap=self.compilation_config["critic"],
        )
        self.gae.set_keys(value=("agents", "state_value"), advantage=("agents", "advantage"),
                          value_target=("agents", "value_target"), reward="team_reward",
                          done=("learning", "done"), terminated=("learning", "terminated"))

    @torch.no_grad()
    def actor_actions(self, observations, agent_ids, *, deterministic=False, return_raw=False):
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

    def update(self, rollout):
        return self.update_rollouts([rollout])

    def update_rollouts(self, rollouts):
        self._require_lora_source()
        fragments = []
        region = getattr(self, "profile_region", None)
        for rollout in rollouts:
            if not rollout.batch_size.numel():
                continue
            with region("rollout_preparation") if region else nullcontext():
                profile = getattr(self, "profile_call", None)
                payload = getattr(self, "profile_payload", None)
                if payload:
                    payload("cpu_to_cuda", rollout, destination=self.device)
                if profile:
                    moving = any(value.device != self.device for value in rollout.values(True, True)
                                 if isinstance(value, torch.Tensor))
                    category = "rollout_device_transfer" if moving else "tensor_device_conversion"
                    fragment = profile(category, rollout.to, self.device, device=self.device)
                    with region("learner_rollout_clone") if region else nullcontext():
                        fragment = profile("tensor_copy", fragment.clone, device=self.device)
                else:
                    fragment = rollout.to(self.device).clone()
                with torch.no_grad():
                    self.gae(fragment)
            fragments.append(fragment)
        if not fragments:
            return {}
        with region("pending_rollout_concatenation") if region else nullcontext():
            data = torch.cat(fragments, dim=0)
        # Keep retired slots in GAE for continuing team credit, then select only
        # actual decisions before advantage normalization and minibatch sampling.
        with region("mappo_active_state_preparation") if region else nullcontext():
            mask = data["agents", "active"].reshape(-1)
            states = data["state"].unsqueeze(-2).expand(-1, len(self.agent_ids), -1).reshape(-1, self.global_state_dim)
            selected = data["agents"].reshape(-1)[mask]
            # TensorDict.numel() clamps empty batches to one in TensorDict 0.14.
            n_samples = selected.batch_size.numel()
            if not n_samples:
                return {}
            samples = TensorDict({"agents": selected.unsqueeze(-1), "state": states[mask]}, [n_samples])
        profile = getattr(self, "profile_call", None)
        metrics = (profile("optimization", optimize_ppo, self, samples, group="agents", device=self.device)
                   if profile else optimize_ppo(self, samples, group="agents"))
        metrics["train/rollout_steps"] = data.numel()
        metrics["train/rollout_agent_samples"] = n_samples
        return metrics
