"""Independent Gaussian actors with explicit routing by learner identity."""
from copy import deepcopy

import torch
from torch import nn

from agents.common.distributions import SquashedGaussianActor


class IndependentActors(SquashedGaussianActor):
    """One complete actor, including exploration parameters, per learner.

    Copies start identically to make full training comparable to per-agent
    adapters initialized from the same driving policy. Parameters never alias.
    The routing keyword matches LoRA's interface for collection and PPO scoring.
    """

    def __init__(self, base, agent_ids):
        super().__init__()
        self.action_dim = base.action_dim
        self.agent_ids = tuple(agent_ids)
        self.actors = nn.ModuleDict({aid: deepcopy(base) for aid in agent_ids})

    def forward(self, obs, adapter_indices=None):
        if (adapter_indices is None or adapter_indices.shape != (len(obs),)
                or adapter_indices.dtype != torch.long):
            raise ValueError('Independent actors require one integer actor index per observation')
        if torch.any((adapter_indices < 0) | (adapter_indices >= len(self.agent_ids))):
            raise ValueError('Invalid independent actor index')
        mean = obs.new_zeros((len(obs), self.action_dim))
        std = obs.new_zeros((len(obs), self.action_dim))
        for index, actor in enumerate(self.actors.values()):
            rows = torch.nonzero(adapter_indices == index, as_tuple=True)[0]
            if rows.numel():
                m, s = actor(obs[rows])
                mean = mean.index_copy(0, rows, m)
                std = std.index_copy(0, rows, s.expand_as(m))
        return mean, std
