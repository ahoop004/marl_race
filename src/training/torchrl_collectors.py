from __future__ import annotations

import pickle

import numpy as np
import torch
from tensordict import TensorDict

from training.rollout_storage import MAPPORolloutStorage


def serialize_rollout(rollout):
    # Pipe.send uses PyTorch's shared-memory reduction for bare tensors, opening
    # a socket and descriptor per storage. A byte payload owns its CPU data and
    # remains readable after a worker exits, including in restricted runtimes.
    return pickle.dumps(rollout, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_rollout(payload):
    return pickle.loads(payload)


class _DecisionBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.steps = []

    def clear(self):
        self.steps.clear()

    def is_full(self):
        return len(self.steps) >= self.capacity

    def add(self, obs, action, reward, log_prob, value, *, terminated,
            truncated, raw_action, next_observation):
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


class PPOCollectorState:
    """CPU decision storage and parent control; workers own no policy networks."""

    def __init__(self, n_steps):
        self.buffer = _DecisionBuffer(n_steps)
        self.should_stop = False

    def pack_rollout(self):
        return torch.stack(self.buffer.steps)

    def apply_reply(self, reply):
        if "collector_control" not in reply:
            return reply
        control = reply["collector_control"]
        self.should_stop = bool(control["stop"])
        return reply["metrics"]


class MAPPOCollectorState(MAPPORolloutStorage):
    """CPU rollout storage and worker counters; inference stays in the parent."""

    def __init__(self, contract, horizon):
        for name, value in contract.items():
            setattr(self, name, value)
        super().__init__(self.agent_ids, self.obs_dims, self.action_dim, horizon)
        self.fragments = []
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
