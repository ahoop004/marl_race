"""Categorical PPO pieces compatible with the existing optimizer and collectors."""
import torch
from torch import nn
from torch.distributions import Categorical, Independent
from agents.common.networks import build_head, model_spec


class SkillDistribution(Independent):
    def __init__(self, logits):
        super().__init__(Categorical(logits=logits), 1)


class CategoricalActor(nn.Module):
    categorical = True
    def __init__(self, obs_dim, n_skills, network):
        super().__init__()
        self.net = build_head(obs_dim, n_skills, model_spec(network, 'actor'))

    def forward(self, observation):
        return (self.net(observation).unsqueeze(-2),)

    def get_action(self, observation, deterministic=False, *, return_raw=False):
        logits, = self(observation)
        dist = SkillDistribution(logits)
        action = dist.mode if deterministic else dist.sample()
        output = (action, dist.log_prob(action))
        return (*output, action.clone()) if return_raw else output

    def entropy_from_parameters(self, logits):
        return SkillDistribution(logits).entropy()


class DurationGAE(nn.Module):
    """Gamma and lambda are in LL decision units; stop traces at every reset."""
    def __init__(self, critic, gamma=.99, lmbda=.95):
        super().__init__()
        self.critic, self.gamma, self.lmbda = critic, gamma, lmbda

    @torch.no_grad()
    def forward(self, data, time_dim=None):
        # Inputs are [time, features] or [trajectory, time, features].
        axis = data.ndim - 1 if time_dim is None else time_dim
        value = self.critic(data['observation']).unsqueeze(-1)
        next_value = self.critic(data['next', 'observation']).unsqueeze(-1)
        reward = data['next', 'reward']
        duration = data['steps_to_next_obs']
        if torch.any(duration < 1):
            raise ValueError('Macro duration must be positive')
        terminated = data['next', 'terminated']
        done = data['next', 'done']
        delta = reward + self.gamma ** duration * (~terminated) * next_value - value
        trace = (self.gamma * self.lmbda) ** duration * (~done)
        advantage = torch.zeros_like(delta)
        carry = torch.zeros_like(delta.select(axis, 0))
        for i in range(data.batch_size[axis] - 1, -1, -1):
            carry = delta.select(axis, i) + trace.select(axis, i) * carry
            advantage.select(axis, i).copy_(carry)
        data['state_value'] = value
        data['advantage'] = advantage
        data['value_target'] = advantage + value
        return data
