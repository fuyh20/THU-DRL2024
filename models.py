import torch
import torch.nn as nn
from utils import mlp
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, action_space, hidden_dims = [400, 300],
                 output_activation=nn.Tanh, activation=nn.ELU, actor_dropout=0.0):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.fcs = mlp(num_states, hidden_dims, num_actions, output_activation=output_activation,
                       activation=activation, dropout_ratio=actor_dropout)
    
    def _normalize(self, action):
        return (action + 1) * (self.action_space.high - self.action_space.low) / 2 + self.action_space.low
    
    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        return super().to(device)

    def forward(self, x):
        # use tanh as output activation
        return self._normalize(self.fcs(x))


class SoftActor(Actor):
    def __init__(self, num_states, num_actions, hidden_size, action_space, log_std_min, log_std_max, activation=nn.ELU, actor_dropout=0.0):
        super().__init__(num_states, num_actions * 2, action_space, hidden_dims=hidden_size, output_activation=nn.Identity, activation=activation, actor_dropout=actor_dropout)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        mean, log_std = self.fcs(state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, sample=True):
        mean, log_std = self.forward(state)
        if not sample:
            return self._normalize(torch.tanh(mean)), None
        std = log_std.exp()
        # dist = TransformedDistribution(Normal(mean, std), TanhTransform(cache_size=1))
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = (dist.log_prob(action) - torch.log(1. - torch.tanh(action).pow(2) + 1e-6)).sum(dim=-1)
        # TODO: some minus np.log(action_scale) here, why?
        return self._normalize(torch.tanh(action)), log_prob


class PPOActor(Actor):
    def __init__(self, num_states, num_actions, hidden_size, action_space, activation=nn.ELU,
                 actor_dropout=0):
        super().__init__(num_states, num_actions, action_space, hidden_size, activation=activation,
                         output_activation=nn.Identity, actor_dropout=actor_dropout)

        self.log_std = nn.Parameter(torch.zeros(1, num_actions))

    def evaluate_actions(self, state, action):
        mean = self.fcs(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    
    def forward(self, state, sample=True):
        "return action, log_prob, entropy"
        mean = self.fcs(state)
        if not sample:
            return mean, None
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dims, output_activation=nn.Identity,
                 activation=nn.ELU, critic_dropout=0.0):
        super().__init__()
        self.fcs = mlp(num_states + num_actions, hidden_dims, 1,
                       output_activation=output_activation,
                       activation=activation, dropout_ratio=critic_dropout)

    def forward(self, state, action):
        return self.fcs(torch.cat([state, action], dim=1)).squeeze()


class ValueNet(Critic):
    def __init__(self, num_states, hidden_dims, activation=nn.ELU, critic_dropout=0.0):
        super().__init__(num_states, 0, hidden_dims, output_activation=nn.Identity,
                         activation=activation, critic_dropout=critic_dropout)

    def forward(self, state):
        return self.fcs(state).squeeze()