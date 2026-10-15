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
from agents.torchrl_rollouts import rollout_groups, transfer_rollout_group, batch_gae
from agents.torchrl_updates import MAPPO_OPTIMIZATION_KEYS, optimize_ppo


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
    # Shared critic inputs at both endpoints and joint-team signals must survive
    # for every agent slot, including retired agents, until MultiAgentGAE ends.
    _ROLLOUT_KEYS = (
        "state", ("agents", "observation"), ("agents", "index"), ("agents", "active"),
        ("agents", "action"), ("agents", "raw_action"), ("agents", "raw_log_prob"),
        ("next", "state"), ("next", "agents", "index"), ("next", "team_reward"),
        ("next", "learning", "done"), ("next", "learning", "terminated"),
        ("next", "learning", "truncated"), ("agents", "state_value"),
        ("next", "agents", "state_value"), "steps_to_next_obs", "is_init", ("next", "is_init"),
        "shifted_valid",
    )

    def select_rollout(self, rollout, *, retain_episode=False):
        """Drop reporting/physical state without copying tensors or agent slots."""
        keys = self._ROLLOUT_KEYS + (("environment_episode",) if retain_episode else ())
        return rollout.select(*keys, strict=False)

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
        all_active = True
        region = getattr(self, "profile_region", None)
        with region("rollout_preparation") if region else nullcontext():
            for group in rollout_groups(self, rollouts):
                data, group_active = transfer_rollout_group(self, group)
                all_active = all_active and group_active
                batch_gae(self, data, len(group))
                with region("post_gae_flatten_mask") if region else nullcontext():
                    fragments.append(data.select(*MAPPO_OPTIMIZATION_KEYS, ("agents", "active")).reshape(-1))
        if not fragments:
            if getattr(self, "distributed", None):
                metrics = optimize_ppo(self, None, group="agents")
                if metrics:
                    metrics.update({"train/rollout_steps": 0, "train/rollout_agent_samples": 0})
                return metrics
            return {}
        with region("pending_rollout_concatenation") if region else nullcontext():
            data = torch.cat(fragments, dim=0)
        # Keep retired slots through GAE, then compute active positions once
        # globally. Per-group CUDA nonzero calls would add host synchronization.
        with region("mappo_active_state_preparation") if region else nullcontext():
            active = data["agents", "active"].reshape(-1)
            selected = data["agents"].exclude("active").reshape(-1)
            if all_active:
                states = data["state"].repeat_interleave(len(self.agent_ids), dim=0)
            else:
                positions = active.nonzero().flatten()
                selected = selected[positions]
                # Gather just the needed states; do not materialize a full
                # time x agent x state expansion before selecting active rows.
                rows = positions.div(len(self.agent_ids), rounding_mode="floor")
                states = data["state"].index_select(0, rows)
            n_samples = selected.batch_size.numel()
            if not n_samples and not getattr(self, "distributed", None):
                return {}
            samples = TensorDict({"agents": selected.unsqueeze(-1), "state": states}, [n_samples])
        profile = getattr(self, "profile_call", None)
        metrics = (profile("optimization", optimize_ppo, self, samples, group="agents", device=self.device)
                   if profile else optimize_ppo(self, samples, group="agents"))
        if not metrics and getattr(self, "distributed", None):
            return {}
        metrics["train/rollout_steps"] = data.numel()
        metrics["train/rollout_agent_samples"] = n_samples
        return metrics
