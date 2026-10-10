"""Architecture-independent squashed Gaussian sampling and likelihoods."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SquashedGaussianActor(nn.Module):
    """forward(obs, adapter_indices=None) returns Gaussian mean and scale.

    Scale may be an action vector or a batch matching mean. Routing wrappers
    use adapter_indices; unshared networks need only implement forward(obs).
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self):
        super().__init__()
        # Deterministic quadrature for E[log |d tanh(z)/dz|]. Nonpersistent
        # buffers preserve the parameter layout of existing checkpoints.
        nodes, weights = np.polynomial.hermite.hermgauss(32)
        self.register_buffer("_entropy_nodes", torch.tensor(nodes * np.sqrt(2), dtype=torch.float32), persistent=False)
        self.register_buffer("_entropy_weights", torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float32), persistent=False)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False, *, return_raw: bool = False,
        adapter_indices: Optional[torch.Tensor] = None,
    ):
        mean, std = self(obs) if adapter_indices is None else self(obs, adapter_indices=adapter_indices)
        if deterministic:
            raw = mean
            action = torch.tanh(mean)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()
            action = torch.tanh(raw)
            log_prob = self._log_prob(dist, raw)
        return (action, log_prob, raw) if return_raw else (action, log_prob)

    @staticmethod
    def _log_jacobian(raw: torch.Tensor) -> torch.Tensor:
        # log(1 - tanh(z)^2), without cancellation at large |z|.
        return 2.0 * (np.log(2.0) - raw - F.softplus(-2.0 * raw))

    def _log_prob(self, dist, raw):
        return (dist.log_prob(raw) - self._log_jacobian(raw)).sum(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        raw_actions: Optional[torch.Tensor] = None,
        *, adapter_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities for already-sampled squashed actions.

        PPO must compare the current policy probability of each rollout action
        with the probability recorded when that same action was collected.
        Training retains pre-tanh samples because float32 tanh loses their
        identity near the bounds. The inverse fallback supports callers that
        only have nonsaturated actions; it cannot recover saturated samples.
        """
        mean, std = self(obs) if adapter_indices is None else self(obs, adapter_indices=adapter_indices)
        dist = torch.distributions.Normal(mean, std)
        bounded_actions = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        recovered = torch.atanh(bounded_actions)
        raw_actions = recovered if raw_actions is None else torch.where(
            torch.isfinite(raw_actions), raw_actions, recovered
        )
        log_prob = self._log_prob(dist, raw_actions)
        samples = mean.unsqueeze(-1) + std.unsqueeze(-1) * self._entropy_nodes
        correction = (self._log_jacobian(samples) * self._entropy_weights).sum(-1)
        entropy = (dist.entropy() + correction).sum(-1)
        return log_prob, entropy

