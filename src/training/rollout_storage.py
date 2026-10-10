"""Native MAPPO transition storage for serial learner updates."""
import numpy as np
import torch
from tensordict import TensorDict
from agents.common.observations import pack_observations


class _DecisionCount:
    def __init__(self):
        self.count = 0

    def size(self):
        return self.count


class MAPPORolloutStorage:
    """Own decision fragments without a policy network, optimizer or GAE."""

    def __init__(self, agent_ids, obs_dims, action_dim, horizon, device="cpu"):
        self.agent_ids = list(agent_ids)
        self.obs_dims = dict(obs_dims)
        self.obs_dim = max(self.obs_dims.values())
        self.action_dim, self.n_steps = action_dim, horizon
        self.device = torch.device(device)
        self._agent_index = {aid: i for i, aid in enumerate(self.agent_ids)}
        self._steps = []
        self.buffers = {aid: _DecisionCount() for aid in self.agent_ids}

    def _validate_agent_batch(self, agent_ids):
        ids = list(agent_ids)
        if len(set(ids)) != len(ids) or any(aid not in self._agent_index for aid in ids):
            raise ValueError("Decision batches require unique, known policy agent IDs")
        return ids

    def pack_observations(self, agent_ids, observations):
        return pack_observations(agent_ids, observations, self.obs_dims, self.obs_dim)

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

    def clear_buffers(self):
        self._steps.clear()
        for buffer in self.buffers.values():
            buffer.count = 0

