"""Action distributions and coefficient transforms for residual composition.

Gaussian PPO actions are latent vectors and use joint latent Normal likelihoods
and entropy. The logistic-normal reference logit is the last skill, so its five
learned logits correspond to skills 0 through 4 in manifest order.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Dirichlet, Independent, Normal

from agents.common.networks import build_head, model_spec


class CompositionActor(nn.Module):
    def __init__(self, obs_dim, n_skills, network, config):
        super().__init__()
        if n_skills != 6:
            raise ValueError('Composition requires the base and five specialists')
        self.kind = config['distribution']
        defaults = {'dirichlet': (0.3, None), 'logistic_normal': (0., 1.),
                    'positive_gaussian': (-1., 0.5), 'signed_gaussian': (0., 0.5)}
        if self.kind not in defaults:
            raise ValueError('Unknown composition distribution')
        self.temperature = float(config.get('temperature', 1.))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Composition temperature must be finite and positive')
        self.action_dim = 6 if self.kind == 'dirichlet' else 5
        self.net = build_head(obs_dim, self.action_dim, model_spec(network, 'actor'))
        # Zero the final weights so the requested distribution holds for every
        # observation at initialization, independent of preceding activations.
        output = [module for module in self.net.modules() if isinstance(module, nn.Linear)][-1]
        nn.init.zeros_(output.weight)
        initial, std = defaults[self.kind]
        if self.kind == 'dirichlet':
            initial = float(config.get('initial_concentration', initial))
            if not math.isfinite(initial) or not 1e-4 < initial < 100.:
                raise ValueError('Initial concentration must lie in (1e-4, 100)')
            bias = math.log(math.expm1(initial - 1e-4))
        else:
            bias = float(config.get('initial_mean', initial))
            std = float(config.get('initial_std', std))
            if not math.isfinite(bias) or not math.isfinite(std) or not math.exp(-5) <= std <= math.exp(2):
                raise ValueError('Invalid initial Gaussian parameters')
            self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(std)))
        nn.init.constant_(output.bias, bias)

    def forward(self, observation):
        values = self.net(observation)
        if self.kind == 'dirichlet':
            return (F.softplus(values).clamp(max=100.) + 1e-4,)
        return values, self.log_std.clamp(-5., 2.).exp().expand_as(values)

    def distribution(self, *parameters):
        if self.kind == 'dirichlet':
            return Dirichlet(parameters[0])
        return Independent(Normal(*parameters), 1)

    def entropy_from_parameters(self, *parameters):
        return self.distribution(*parameters).entropy()

    def representative(self, *parameters):
        if self.kind == 'dirichlet':
            return parameters[0] / parameters[0].sum(-1, keepdim=True)
        return parameters[0]

    def coefficients(self, action, *, temperature=None):
        if action.ndim == 0 or action.shape[-1] != self.action_dim or not torch.isfinite(action).all():
            raise ValueError('Invalid composition action')
        if temperature is not None and self.kind != 'logistic_normal':
            raise ValueError('Temperature applies only to logistic-normal composition')
        if self.kind == 'dirichlet':
            if torch.any(action < 0) or not torch.allclose(action.sum(-1), torch.ones_like(action[..., 0]), atol=1e-5, rtol=1e-5):
                raise ValueError('Dirichlet actions must lie on the simplex')
            return action
        if self.kind == 'logistic_normal':
            temp = self.temperature if temperature is None else float(temperature)
            if not math.isfinite(temp) or temp <= 0:
                raise ValueError('Inference temperature must be finite and positive')
            return torch.softmax(torch.cat((action, torch.zeros_like(action[..., :1])), -1) / temp, -1)
        result = torch.tanh(action)
        return (result + 1.) / 2. if self.kind == 'positive_gaussian' else result

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
            convex = self.kind in ('dirichlet', 'logistic_normal')
            indices = (coefficients if convex else coefficients.abs()).topk(top_k, dim=-1).indices
            sparse = torch.zeros_like(coefficients).scatter(-1, indices, coefficients.gather(-1, indices))
            coefficients = sparse / sparse.sum(-1, keepdim=True) if convex else sparse
        return coefficients
