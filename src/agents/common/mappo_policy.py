"""Routed MAPPO policy setup, inference, transfer and checkpoint contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim

from agents.common.networks import Actor, Critic
from agents.common.lora import LoRAActor, resolve_lora_config
from agents.common.independent import IndependentActors
from agents.common.observations import pack_observations
from utils.torch_io import resolve_device


class MAPPOPolicy:
    """Local actors and a centralized critic without rollout or update code."""

    def __init__(
        self,
        obs_dim: int,
        global_state_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        agent_ids: List[str],
        params: Dict,
    ) -> None:
        if not agent_ids:
            raise ValueError("MAPPO requires at least one trainable agent ID.")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("MAPPO trainable agent IDs must be unique and ordered.")
        self.obs_dims = dict(params.get("_observation_dims") or {aid: obs_dim for aid in agent_ids})
        if set(self.obs_dims) != set(agent_ids) or any(
                isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in self.obs_dims.values()):
            raise ValueError("MAPPO observation dimensions must name every learner with a positive width")
        self.obs_dim = obs_dim = max(self.obs_dims.values())
        self.global_state_dim = global_state_dim
        self.global_state_contract_version = str(
            params.get("_global_state_contract_version", "legacy_unspecified")
        )
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = len(self.action_low)
        self.action_contract = dict(params.get("_action_contract", {"speed_control": "direct"}))
        self.physics_contract = params.get("_physics_contract")
        self.observation_contract = params.get("_observation_contract")
        self.observation_contracts = dict(params.get("_observation_contracts") or {
            aid: self.observation_contract for aid in agent_ids})
        if set(self.observation_contracts) != set(agent_ids):
            raise ValueError("Observation contracts must name every learner")
        self.pretrained_actor_observation_extension = params.get("pretrained_actor_observation_extension")
        self.agent_ids = list(agent_ids)
        self._agent_index = {aid: idx for idx, aid in enumerate(self.agent_ids)}
        self.actor_mode = str(params.get("actor_mode", "shared"))
        if self.actor_mode not in {"shared", "independent"}:
            raise ValueError("actor_mode must be shared or independent")
        self.lora_config = resolve_lora_config(params.get("lora"))
        if len(set(self.obs_dims.values())) > 1 and not (
                self.lora_config and self.lora_config["mode"] == "per_agent"):
            raise ValueError("Different observation dimensions require per_agent LoRA")
        if self.actor_mode == "independent" and self.lora_config is not None:
            raise ValueError("Independent actors cannot also use LoRA; use shared with per_agent adapters")
        self.pretrained_actor_source = None
        self._lora_ready = self.lora_config is None
        self.last_raw_actions: Dict[str, np.ndarray] = {}

        self.critic_mode = str(params.get("critic_mode", "agent_conditioned")).strip().lower()
        if self.critic_mode not in {"shared_team", "agent_conditioned"}:
            raise ValueError(
                "MAPPO critic_mode must be 'shared_team' or 'agent_conditioned', "
                f"got {self.critic_mode!r}."
            )
        self.reward_mode = str(params.get("reward_mode", "individual")).strip().lower()
        self.team_return_mode = str(params.get("team_return_mode", "per_agent"))
        if self.team_return_mode not in {"per_agent", "joint"}:
            raise ValueError("team_return_mode must be per_agent or joint")
        if self.team_return_mode == "joint" and (
            self.reward_mode != "team_shared" or self.critic_mode != "shared_team"
        ):
            raise ValueError("Joint team returns require team_shared rewards and shared_team critic")
        self.team_reward_reduction = str(
            params.get("team_reward_reduction", "mean")
        ).strip().lower()
        if self.reward_mode not in {"individual", "team_shared"}:
            raise ValueError(
                "MAPPO reward_mode must be 'individual' or 'team_shared', "
                f"got {self.reward_mode!r}."
            )
        if self.team_reward_reduction not in {"mean", "sum"}:
            raise ValueError(
                "MAPPO team_reward_reduction must be 'mean' or 'sum', "
                f"got {self.team_reward_reduction!r}."
            )
        if self.reward_mode == "individual" and self.critic_mode == "shared_team":
            raise ValueError(
                "MAPPO individual rewards require critic_mode='agent_conditioned'."
            )

        # Hyperparameters
        self.lr = float(params.get("learning_rate", 3e-4))
        self.gamma = float(params.get("gamma", 0.99))
        self.gae_lambda = float(params.get("gae_lambda", 0.95))
        self.clip_range = float(params.get("clip_range", 0.2))
        self.ent_coef = float(params.get("ent_coef", 0.01))
        self.vf_coef = float(params.get("vf_coef", 0.5))
        self.max_grad_norm = float(params.get("max_grad_norm", 0.5))
        self.n_steps = int(params.get("n_steps", 2048))
        self.n_epochs = int(params.get("n_epochs", 10))
        self.batch_size = int(params.get("batch_size", 64))

        hidden_dims: List[int] = list(
            params.get("pi_hidden_dims", params.get("hidden_dims", [256, 256]))
        )
        vf_dims: List[int] = list(
            params.get("vf_hidden_dims", params.get("hidden_dims", [256, 256]))
        )
        activation: str = str(params.get("activation", "tanh"))
        self.actor_hidden_dims = list(hidden_dims)
        self.critic_hidden_dims = list(vf_dims)
        self.activation = activation

        device_str = str(params.get("device", "cpu"))
        self.device = resolve_device([device_str])

        # Shared actor (local obs → action)
        self.actor = Actor(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)

        # The team critic estimates one shared V(s).  The agent-conditioned
        # critic estimates V_i(s) by appending a focal-agent one-hot vector.
        # In both cases the actor remains decentralized and sees local obs only.
        self.critic_input_dim = global_state_dim + (
            len(self.agent_ids) if self.critic_mode == "agent_conditioned" else 0
        )
        self.critic = Critic(self.critic_input_dim, vf_dims, activation).to(self.device)
        self._agent_identity = torch.eye(
            len(self.agent_ids), dtype=torch.float32, device=self.device
        )

        # Build adapters after the critic so its initialization matches the
        # full-fine-tuning control under the same seed.
        if self.actor_mode == "independent":
            self.actor = IndependentActors(self.actor, self.agent_ids).to(self.device)
        self.lora_contract = None
        if self.lora_config is not None:
            inputs = ([self.obs_dims[aid] for aid in self.agent_ids]
                      if self.lora_config["mode"] == "per_agent" else None)
            self.actor = LoRAActor(self.actor, self.lora_config, len(self.agent_ids), inputs).to(self.device)
            self.lora_contract = {
                "version": 1, **self.lora_config,
                "target_layers": self.actor.target_layers,
                "agent_to_adapter": {aid: (i if self.lora_config["mode"] == "per_agent" else 0)
                                     for i, aid in enumerate(self.agent_ids)},
            }
            if len(set(self.obs_dims.values())) > 1:
                self.lora_contract["observation_dims"] = dict(self.obs_dims)
        self._optim_parameters = tuple(p for p in self.actor.parameters() if p.requires_grad) + tuple(
            self.critic.parameters())
        self.optimizer = optim.Adam(self._optim_parameters, lr=self.lr)

    @property
    def per_agent_adapters(self) -> bool:
        return self.lora_config is not None and self.lora_config["mode"] == "per_agent"

    @property
    def routed_actor(self) -> bool:
        return self.actor_mode == "independent" or self.per_agent_adapters

    def _require_lora_source(self):
        if not self._lora_ready:
            raise ValueError("LoRA requires a pretrained PPO actor or a matching MAPPO checkpoint before use")

    def actor_actions(self, observations, agent_ids, *, deterministic=False, return_raw=False):
        """Route rows, including repeated IDs from independent environments."""
        self._require_lora_source()
        if len(agent_ids) != len(observations) or any(aid not in self._agent_index for aid in agent_ids):
            raise ValueError("Actor rows require matching, known agent IDs")
        kwargs = {}
        if self.routed_actor:
            kwargs["adapter_indices"] = torch.tensor(
                [self._agent_index[aid] for aid in agent_ids], device=self.device, dtype=torch.long)
        return self.actor.get_action(observations, deterministic=deterministic,
                                     return_raw=return_raw, **kwargs)

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False, *, agent_id: Optional[str] = None
    ) -> Tuple[np.ndarray, float]:
        """Sample a local action; routed policies require the agent ID.

        Returns
        -------
        action_normalized : np.ndarray
            Action in ``[-1, 1]`` — caller denormalizes for ``env.step()``.
        log_prob : float
            Log probability of the sampled action.
        """
        if self.routed_actor and agent_id is None:
            raise ValueError("Routed actor act requires agent_id")
        aid = agent_id if agent_id is not None else self.agent_ids[0]
        obs_t = torch.as_tensor(self.pack_observations([aid], [obs]), dtype=torch.float32, device=self.device)
        action_t, log_prob_t = self.actor_actions(
            obs_t, [agent_id if agent_id is not None else self.agent_ids[0]], deterministic=deterministic)
        return (
            action_t.squeeze(0).cpu().numpy(),
            float(log_prob_t.squeeze()),
        )

    @torch.no_grad()
    def act_batch(
        self,
        agent_ids: Sequence[str],
        observations: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Select actions for an ordered active-agent batch with one actor call."""
        ordered_ids = self._validate_agent_batch(agent_ids)
        if not ordered_ids:
            return {}, {}
        obs = self.pack_observations(ordered_ids, observations)
        if obs.shape != (len(ordered_ids), self.obs_dim):
            raise ValueError(
                "Expected batched local observations with shape "
                f"({len(ordered_ids)}, {self.obs_dim}), got {obs.shape}."
            )
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_t, log_prob_t, raw_t = self.actor_actions(
            obs_t, ordered_ids, deterministic=deterministic, return_raw=True
        )
        # One device-to-host transfer for the complete joint decision.
        result = torch.cat((action_t, log_prob_t.unsqueeze(-1), raw_t), dim=-1).cpu().numpy()
        self.last_raw_actions = {
            aid: result[index, self.action_dim + 1:].copy()
            for index, aid in enumerate(ordered_ids)
        }
        actions = {
            agent_id: result[index, : self.action_dim].copy()
            for index, agent_id in enumerate(ordered_ids)
        }
        log_probs = {
            agent_id: float(result[index, self.action_dim])
            for index, agent_id in enumerate(ordered_ids)
        }
        return actions, log_probs

    def pack_observations(self, agent_ids, observations):
        return pack_observations(agent_ids, observations, self.obs_dims, self.obs_dim)

    def _validate_agent_batch(self, agent_ids: Sequence[str]) -> List[str]:
        ordered_ids = [str(agent_id) for agent_id in agent_ids]
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("MAPPO inference batches cannot contain duplicate agent IDs.")
        unknown = [agent_id for agent_id in ordered_ids if agent_id not in self._agent_index]
        if unknown:
            raise ValueError(f"MAPPO inference batch contains unknown agent IDs: {unknown}.")
        return ordered_ids

    def _critic_input(
        self,
        global_state: np.ndarray,
        agent_id: Optional[str] = None,
    ) -> np.ndarray:
        state = np.asarray(global_state, dtype=np.float32).reshape(-1)
        if state.size != self.global_state_dim:
            raise ValueError(
                f"Expected global state dimension {self.global_state_dim}, got {state.size}."
            )
        if self.critic_mode == "shared_team":
            return state
        if agent_id not in self._agent_index:
            raise ValueError(
                "agent_conditioned critic requires a known agent_id; "
                f"got {agent_id!r}."
            )
        identity = np.zeros(len(self.agent_ids), dtype=np.float32)
        identity[self._agent_index[agent_id]] = 1.0
        return np.concatenate((state, identity))

    @torch.no_grad()
    def evaluate_state(
        self,
        global_state: np.ndarray,
        agent_id: Optional[str] = None,
    ) -> float:
        """Estimate value of a global state using the centralized critic.

        Parameters
        ----------
        global_state:
            Flat numpy array from ``env.get_global_state().vector``.
        """
        critic_input = self._critic_input(global_state, agent_id)
        # GlobalState vectors are intentionally read-only. Copy into owned
        # tensor storage rather than aliasing immutable NumPy memory.
        gs_t = torch.tensor(
            critic_input, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return float(self.critic(gs_t).squeeze())

    @torch.no_grad()
    def evaluate_states(
        self,
        global_state: np.ndarray,
        agent_ids: Sequence[str],
    ) -> Dict[str, float]:
        """Estimate ordered per-agent centralized values with one critic call."""
        ordered_ids = self._validate_agent_batch(agent_ids)
        if not ordered_ids:
            return {}
        state = np.asarray(global_state, dtype=np.float32).reshape(-1)
        if state.size != self.global_state_dim:
            raise ValueError(
                f"Expected global state dimension {self.global_state_dim}, got {state.size}."
            )
        # GlobalState vectors are intentionally read-only. Copy into owned
        # tensor storage rather than aliasing immutable NumPy memory.
        states_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(len(ordered_ids), -1)
        if self.critic_mode == "agent_conditioned":
            indices = torch.as_tensor(
                [self._agent_index[agent_id] for agent_id in ordered_ids],
                dtype=torch.long,
                device=self.device,
            )
            identities = self._agent_identity.index_select(0, indices)
            states_t = torch.cat((states_t, identities), dim=-1)
        values = self.critic(states_t).cpu().numpy()
        return {
            agent_id: float(values[index])
            for index, agent_id in enumerate(ordered_ids)
        }

    def load_pretrained_actor(self, path: str) -> None:
        """Initialize actors from checkpoint weights, retaining critic and optimizer."""
        from utils.torch_io import safe_load, validate_checkpoint_compatibility

        ckpt = safe_load(path, map_location=self.device)
        validate_checkpoint_compatibility(ckpt, {
            "algorithm": "ppo",
            "action_dim": self.action_dim,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "actor_hidden_dims": self.actor_hidden_dims,
            "activation": self.activation,
            "physics_contract": self.physics_contract,
            "action_contract": self.action_contract,
        })
        if not isinstance(ckpt.get("actor"), Mapping):
            raise ValueError("Pretrained PPO checkpoint has no actor state")
        source_width = ckpt.get("obs_dim")
        if isinstance(source_width, bool) or not isinstance(source_width, int) or source_width <= 0:
            raise ValueError("Pretrained PPO checkpoint requires a positive obs_dim")
        source_contract = ckpt.get("observation_contract")
        for aid in self.agent_ids:
            width = self.obs_dims[aid]
            destination = self.observation_contracts[aid]
            if source_width == width:
                validate_checkpoint_compatibility(ckpt, {"observation_contract": destination})
            elif source_width < width and self.pretrained_actor_observation_extension == "frenet_neighbors":
                # Only appended neighbor inputs may extend the solo driving prefix.
                from agents.common.observations import observation_layout
                common, driving, total = observation_layout(source_contract)
                dest_common, dest_driving, dest_total = observation_layout(destination)
                if (source_width != driving or total != driving
                        or dest_driving != driving or dest_total != width
                        or not destination["observation"].get("frenet_neighbors", {}).get("enabled")):
                    raise ValueError(f"Unsupported frenet_neighbors observation extension for {aid}")
                validate_checkpoint_compatibility({"observation_prefix": common},
                                                  {"observation_prefix": dest_common})
            else:
                raise ValueError(f"Unsupported pretrained observation width {source_width} → {width} for {aid}; "
                                 "wider inputs require an explicit frenet_neighbors extension")

        recipient = (self.actor.actors[self.agent_ids[0]]
                     if self.actor_mode == "independent" else self.actor)
        expected = (recipient.base_state_dict() if self.lora_config is not None
                    else recipient.state_dict())
        actor_state = dict(ckpt["actor"])
        old_weight = actor_state.get("net.0.weight")
        if (not isinstance(old_weight, torch.Tensor) or old_weight.ndim != 2
                or old_weight.shape != (expected["net.0.weight"].shape[0], source_width)):
            raise ValueError("Pretrained first-layer weights do not match the source observation width")
        if source_width < self.obs_dim:
            expanded = torch.zeros_like(expected["net.0.weight"])
            expanded[:, :source_width] = old_weight
            actor_state["net.0.weight"] = expanded
        validate_checkpoint_compatibility(ckpt, {}, states=[("actor", actor_state, expected)])
        import hashlib
        source = {
            "path": str(Path(path).resolve()),
            "algorithm": "ppo",
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        # All contracts and tensors have passed before any model state changes.
        if self.lora_config is not None:
            self.actor.reset_adapters()
            if self.lora_config.get("per_agent_log_std"):
                actor_state.update({f"log_stds.{i}": actor_state["log_std"].clone()
                                    for i in range(len(self.agent_ids))})
            actor_state = {**self.actor.state_dict(), **actor_state}
        if self.actor_mode == "independent":
            for actor in self.actor.actors.values():
                actor.load_state_dict(actor_state, strict=True)
        else:
            self.actor.load_state_dict(actor_state, strict=True)
        self.pretrained_actor_source = source
        self._lora_ready = True

    def save(self, path: str) -> None:
        self._require_lora_source()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **({"actors": {aid: actor.state_dict() for aid, actor in self.actor.actors.items()}}
                   if self.actor_mode == "independent" else {"actor": self.actor.state_dict()}),
                "actor_mode": self.actor_mode,
                "actor_routing": dict(self._agent_index),
                "advantage_normalization": ("per_agent" if self.routed_actor and self.team_return_mode == "per_agent" else "pooled"),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "algorithm": "mappo",
                "agent_ids": self.agent_ids,
                "obs_dim": self.obs_dim,
                "obs_dims": self.obs_dims,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "action_contract": self.action_contract,
                "physics_contract": self.physics_contract,
                "observation_contract": self.observation_contract,
                "observation_contracts": self.observation_contracts,
                "global_state_dim": self.global_state_dim,
                "global_state_contract_version": self.global_state_contract_version,
                "critic_input_dim": self.critic_input_dim,
                "critic_mode": self.critic_mode,
                "reward_mode": self.reward_mode,
                "team_reward_reduction": self.team_reward_reduction,
                "team_return_mode": self.team_return_mode,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "activation": self.activation,
                "lora_contract": self.lora_contract,
                "pretrained_actor_source": self.pretrained_actor_source,
            },
            path,
        )

    def load(self, path: str) -> None:
        from utils.torch_io import safe_load, validate_checkpoint_compatibility
        ckpt = safe_load(path, map_location=self.device)
        fields = (
            "actor_mode", "agent_ids", "actor_hidden_dims", "critic_hidden_dims",
            "activation", "critic_mode", "global_state_dim", "global_state_contract_version",
            "critic_input_dim", "obs_dim", "obs_dims", "observation_contract",
            "observation_contracts", "action_dim", "action_low", "action_high",
            "action_contract", "physics_contract", "lora_contract", "reward_mode",
            "team_reward_reduction", "team_return_mode",
        )
        validate_checkpoint_compatibility(ckpt, {
            "algorithm": "mappo", "actor_routing": self._agent_index,
            **{field: getattr(self, field) for field in fields},
        })
        if self.actor_mode == "independent":
            actors = ckpt.get("actors")
            if not isinstance(actors, Mapping) or set(actors) != set(self.agent_ids):
                raise ValueError("Checkpoint actors must name every learner")
            if not all(isinstance(state, Mapping) for state in actors.values()):
                raise ValueError("Checkpoint actors must contain complete actor states")
            actor_state = {f"actors.{aid}.{key}": value
                           for aid, state in actors.items() for key, value in state.items()}
        else:
            actor_state = ckpt.get("actor")
        validate_checkpoint_compatibility(ckpt, {}, states=[
            ("actor", actor_state, self.actor.state_dict()),
            ("critic", ckpt.get("critic"), self.critic.state_dict()),
        ])
        self.actor.load_state_dict(actor_state, strict=True)
        self.critic.load_state_dict(ckpt["critic"], strict=True)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.pretrained_actor_source = ckpt.get("pretrained_actor_source")
        self._lora_ready = True

