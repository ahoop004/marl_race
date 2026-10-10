from __future__ import annotations

import pickle

import numpy as np
import torch
from tensordict import TensorDict

from agents.mappo import MAPPOAgent
from agents.torchrl_mappo import TorchRLMAPPOAgent
from training.on_policy_trainer import _RemotePolicy


def serialize_rollout(rollout):
    # Pipe.send uses PyTorch's shared-memory reduction for bare tensors, opening
    # a socket and descriptor per storage. A byte payload owns its CPU data and
    # remains readable after a worker exits, including in restricted runtimes.
    return pickle.dumps(rollout, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_rollout(payload):
    return pickle.loads(payload)


class _DecisionBuffer:
    requires_next_observation = True

    def __init__(self, capacity):
        self.capacity = capacity
        self.steps = []

    def clear(self):
        self.steps.clear()

    def is_full(self):
        return len(self.steps) >= self.capacity

    def add(self, obs, action, reward, log_prob, value, *, terminated,
            truncated, raw_action, next_observation, final_value=None):
        # Own the arrays: task snapshots and policy replies may be reused.
        def tensor(value):
            return torch.tensor(np.asarray(value).copy(), dtype=torch.float32)

        self.steps.append(TensorDict({
            "observation": tensor(obs), "action": tensor(action),
            "raw_action": tensor(raw_action), "raw_log_prob": tensor(log_prob),
            "next": TensorDict({
                "observation": tensor(next_observation), "reward": tensor([reward]),
                "done": torch.tensor([terminated or truncated]),
                "terminated": torch.tensor([terminated]),
            }, []),
        }, []))


class PPOCollectorPolicy(_RemotePolicy):
    def __init__(self, connection, n_steps, obs_dim, action_dim, gamma, gae_lambda, **kwargs):
        super().__init__(connection, n_steps, obs_dim, action_dim, gamma, gae_lambda,
                         buffer=_DecisionBuffer(n_steps), **kwargs)

    def pack_rollout(self, next_value):
        return torch.stack(self.buffer.steps)


class MAPPOCollectorAgent:
    # Workers own no networks. Reuse the serial native transition contract;
    # the learner computes MultiAgentGAE before pooling independent fragments.
    store_batch = TorchRLMAPPOAgent.store_batch
    store_team_step = TorchRLMAPPOAgent.store_team_step
    set_next_state = TorchRLMAPPOAgent.set_next_state
    any_buffer_full = TorchRLMAPPOAgent.any_buffer_full
    clear_buffers = TorchRLMAPPOAgent.clear_buffers
    _validate_agent_batch = MAPPOAgent._validate_agent_batch
    pack_observations = MAPPOAgent.pack_observations

    def __init__(self, contract, horizon):
        for name, value in contract.items():
            setattr(self, name, value)
        self.n_steps = horizon
        self.device = torch.device("cpu")
        self._agent_index = {aid: index for index, aid in enumerate(self.agent_ids)}
        self.buffers = TorchRLMAPPOAgent._make_buffers(self)
        self.fragments = []
        self.last_raw_actions = {}
        self.policy_version = 0
        self.physics_steps_collected = 0

    def finish_fragment(self, next_values):
        if self._steps:
            self.fragments.append(torch.stack(self._steps))
        self.clear_buffers()
        return {}

    def take_fragments(self):
        result, self.fragments = self.fragments, []
        return result
