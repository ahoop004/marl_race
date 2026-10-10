from __future__ import annotations

import pickle

import numpy as np
import torch
from tensordict import TensorDict

from agents.common.mappo_policy import MAPPOPolicy
from agents.torchrl_mappo import TorchRLMAPPOAgent


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


class PPOCollectorPolicy:
    """CPU decision storage and parent control; workers own no policy networks."""

    def __init__(self, n_steps, *, map_scheduler=None, worker_id=0):
        self.buffer = _DecisionBuffer(n_steps)
        self.map_scheduler = map_scheduler
        self.worker_id = worker_id
        self.should_stop = False
        self._training_bundles = None

    def pack_rollout(self):
        return torch.stack(self.buffer.steps)

    def apply_reply(self, reply):
        if "collector_control" not in reply:
            return reply
        control = reply["collector_control"]
        self.should_stop = bool(control["stop"])
        bundles = tuple(control.get("training_bundles", ()))
        if bundles and bundles != self._training_bundles:
            if self.map_scheduler is None:
                raise RuntimeError("Collector received map curriculum without a scheduler")
            offset = self.worker_id % len(bundles)
            self.map_scheduler.set_training_bundles(list(bundles[offset:] + bundles[:offset]))
            self._training_bundles = bundles
        return reply["metrics"]


class MAPPOCollectorAgent:
    # Workers own no networks. Reuse the serial native transition contract;
    # the learner computes MultiAgentGAE before pooling independent fragments.
    store_batch = TorchRLMAPPOAgent.store_batch
    store_team_step = TorchRLMAPPOAgent.store_team_step
    set_next_state = TorchRLMAPPOAgent.set_next_state
    any_buffer_full = TorchRLMAPPOAgent.any_buffer_full
    clear_buffers = TorchRLMAPPOAgent.clear_buffers
    _validate_agent_batch = MAPPOPolicy._validate_agent_batch
    pack_observations = MAPPOPolicy.pack_observations

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
