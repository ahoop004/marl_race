"""Action distributions and coefficient transforms for residual composition.

Gaussian PPO actions are latent vectors and use joint latent Normal likelihoods
and entropy. The logistic-normal reference logit is the last skill, so its five
learned logits correspond to skills 0 through 4 in manifest order.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Dirichlet, Independent, Normal

from agents.common.networks import build_head, model_spec


@dataclass(frozen=True)
class CompositionSpec:
    action_dim: int
    sampled_action: str
    coefficient_constraint: str
    backend: str
    entropy_convention: str
    likelihood: str
    entropy: str
    initial: float = 0.
    std: float | None = None

    @property
    def contract(self):
        return {key: getattr(self, key) for key in (
            'sampled_action', 'coefficient_constraint', 'backend', 'entropy_convention')}

    @property
    def categorical(self):
        return self.sampled_action == 'categorical'

    @property
    def simplex(self):
        return self.sampled_action == 'simplex'

    @property
    def convex(self):
        return self.coefficient_constraint == 'convex'


def _gaussian(constraint, mean, std):
    return CompositionSpec(5, 'gaussian_latent', constraint, 'shared_residual', 'latent_gaussian',
                           'composition_joint_latent_gaussian', 'latent_gaussian_v1', mean, std)


_DISTRIBUTIONS = {
    'categorical': CompositionSpec(1, 'categorical', 'hard_selection', 'categorical_bank',
                                   'categorical_exact', 'categorical_skill_v1', 'categorical_exact_v1'),
    'dirichlet': CompositionSpec(6, 'simplex', 'convex', 'shared_residual', 'dirichlet_differential',
                                 'composition_joint_dirichlet', 'dirichlet_differential_v1', .3),
    'logistic_normal': _gaussian('convex', 0., 1.),
    'positive_gaussian': _gaussian('independent_positive', -1., .5),
    'signed_gaussian': _gaussian('signed', 0., .5),
}


def composition_spec(config):
    kind = config.get('distribution', 'categorical')
    if not isinstance(kind, str) or kind not in _DISTRIBUTIONS:
        raise ValueError(f'Unknown hierarchical action distribution: {kind!r}')
    spec = _DISTRIBUTIONS[kind]
    if any(config.get(key) not in (None, value) for key, value in spec.contract.items()):
        raise ValueError('Composition distribution, action, constraint, entropy and backend contract disagree')
    return spec


def coefficient_transform(action, spec, temperature=1.):
    """Transform sampled actions on the action's current backend."""
    if spec.simplex:
        return action
    tensor = isinstance(action, torch.Tensor)
    if spec.convex:
        zero = torch.zeros_like(action[..., :1]) if tensor else np.zeros_like(action[..., :1])
        if tensor:
            logits = torch.cat((action, zero), dim=-1)
            centered = logits - logits.amax(dim=-1, keepdim=True)
            return torch.softmax(centered / max(temperature, torch.finfo(logits.dtype).tiny), dim=-1)
        logits = np.concatenate((action, zero), axis=-1).astype(np.float64)
        weights = np.exp((logits - logits.max(axis=-1, keepdims=True)) / temperature)
        return (weights / weights.sum(axis=-1, keepdims=True)).astype(np.float32)
    residual = torch.tanh(action) if tensor else np.tanh(action)
    if spec.coefficient_constraint == 'independent_positive':
        residual = (residual + 1.) / 2.
    return residual


class CompositionDirichlet(Dirichlet):
    @property
    def mode(self):
        # A representative action in the interior also has a finite likelihood
        # for the sparse concentrations used at initialization.
        return self.mean

    deterministic_sample = mode


@dataclass(frozen=True)
class CompositionExecution:
    """Evaluation-only execution coefficients, distinct from PPO actions."""
    coefficients: object


class CompositionActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim, n_skills, network, config):
        super().__init__()
        if n_skills != 6:
            raise ValueError('Composition requires the base and five specialists')
        self.kind = config['distribution']
        self.spec = composition_spec(config)
        if self.spec.categorical:
            raise ValueError('Composition requires a continuous distribution')
        if config.get('coefficient_bound', 1.) != 1.:
            raise ValueError('Composition coefficient bound must be 1.0')
        self.temperature = float(config.get('temperature', 1.))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Composition temperature must be finite and positive')
        self.action_dim = self.spec.action_dim
        self.net = build_head(obs_dim, self.action_dim, model_spec(network, 'actor'))
        # Zero the final weights so the requested distribution holds for every
        # observation at initialization, independent of preceding activations.
        output = [module for module in self.net.modules() if isinstance(module, nn.Linear)][-1]
        nn.init.zeros_(output.weight)
        initial, std = self.spec.initial, self.spec.std
        if self.spec.simplex:
            initial = float(config.get('initial_concentration', initial))
            if not math.isfinite(initial) or not 1e-4 < initial < 100.:
                raise ValueError('Initial concentration must lie in (1e-4, 100)')
            bias = math.log(math.expm1(initial - 1e-4))
            self.initialization = {'concentration': initial}
        else:
            bias = float(initial if config.get('initial_mean') is None else config['initial_mean'])
            std = float(std if config.get('initial_std') is None else config['initial_std'])
            if not math.isfinite(bias) or not math.isfinite(std) or not math.exp(-5) <= std <= math.exp(2):
                raise ValueError('Invalid initial Gaussian parameters')
            self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(std)))
            self.initialization = {'latent_mean': bias, 'latent_std': std}
        nn.init.constant_(output.bias, bias)

    def forward(self, observation):
        values = self.net(observation)
        if self.spec.simplex:
            return (F.softplus(values).clamp(max=100.) + 1e-4,)
        return values, self.log_std.clamp(-5., 2.).exp().expand_as(values)

    def distribution(self, *parameters):
        if self.spec.simplex:
            return CompositionDirichlet(parameters[0])
        return Independent(Normal(*parameters), 1)

    def entropy_from_parameters(self, *parameters):
        return self.distribution(*parameters).entropy()

    def representative(self, *parameters):
        if self.spec.simplex:
            return parameters[0] / parameters[0].sum(-1, keepdim=True)
        return parameters[0]

    def coefficients(self, action, *, temperature=None):
        if not self.valid_action(action):
            raise ValueError('Invalid composition action')
        if temperature is not None and self.kind != 'logistic_normal':
            raise ValueError('Temperature applies only to logistic-normal composition')
        temp = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError('Inference temperature must be finite and positive')
        return coefficient_transform(action, self.spec, temp)

    def valid_action(self, action):
        if action.ndim == 0 or action.shape[-1] != self.action_dim:
            raise ValueError('Invalid composition action shape')
        valid = action.isfinite().all()
        if self.spec.simplex:
            total = action.sum(-1)
            valid &= (action >= 0).all() & torch.isclose(
                total, torch.ones_like(total), atol=1e-5, rtol=1e-5).all()
        return valid

    @torch.no_grad()
    def get_action(self, observation, deterministic=False, *, return_raw=False):
        parameters = self(observation)
        dist = self.distribution(*parameters)
        action = self.representative(*parameters) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        if not torch.isfinite(log_prob).all():
            raise ValueError('Nonfinite composition log probability')
        result = (action, log_prob)
        return (*result, action.clone()) if return_raw else result

    @torch.no_grad()
    def inference_coefficients(self, observation, *, strategy='representative', samples=32,
                               temperature=None, top_k=None):
        parameters = self(observation)
        if strategy == 'representative':
            coefficients = self.coefficients(self.representative(*parameters), temperature=temperature)
        elif strategy == 'stochastic':
            coefficients = self.coefficients(self.distribution(*parameters).sample(), temperature=temperature)
        elif strategy == 'monte_carlo':
            if type(samples) is not int or samples < 1:
                raise ValueError('Monte Carlo sample count must be positive')
            coefficients = self.coefficients(self.distribution(*parameters).sample((samples,)), temperature=temperature).mean(0)
        else:
            raise ValueError('Unknown composition inference strategy')
        if top_k is not None:
            if top_k not in (1, 2):
                raise ValueError('Sparse inference supports top one or two')
            convex = self.spec.convex
            indices = (coefficients if convex else coefficients.abs()).topk(top_k, dim=-1).indices
            sparse = torch.zeros_like(coefficients).scatter(-1, indices, coefficients.gather(-1, indices))
            coefficients = sparse / sparse.sum(-1, keepdim=True) if convex else sparse
        return coefficients
