import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.transforms import TanhTransform
from torch.distributions import Normal, TransformedDistribution

from typing import Optional
from jaxtyping import Float, jaxtyped
from beartype import beartype

def mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, activation=nn.ELU):
    sizes = [input_size] + list(layer_sizes) + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, action_space, hidden_dims = [400, 300], output_activation=nn.Tanh):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.fcs = mlp(num_states, hidden_dims, num_actions, output_activation=output_activation)
    
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
    def __init__(self, num_states, num_actions, hidden_size, action_space, log_std_min, log_std_max):
        super().__init__(num_states, num_actions * 2, action_space, hidden_dims=hidden_size, output_activation=nn.Identity)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    @jaxtyped(typechecker=beartype)
    def forward(self, 
            state: Float[Tensor, "*batch_size state_dim"]
        ) -> tuple[Float[Tensor, "*batch_size action_dim"], Float[Tensor, "*batch_size action_dim"]]:
        """
        Obtain mean and log(std) from the fully-connected network.
        Crop the value of log_std to the specified range.
        """

        ############################
        # YOUR IMPLEMENTATION HERE #)
        mean_log_std = self.fcs(state)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        ############################
        return mean, log_std

    @jaxtyped(typechecker=beartype)
    def evaluate(self, 
            state: Float[Tensor, "*batch_size state_dim"],
            sample: bool = True
        ) -> tuple[Float[Tensor, "*batch_size action_dim"], Optional[Float[Tensor, "*batch_size"]]]:
        
        mean, log_std = self.forward(state)
        if not sample:
            return self._normalize(torch.tanh(mean)), None
        
        # sample action from N(mean, std) if sample is True
        # obtain log_prob for policy and Q function update
        # Hint: remember the reparameterization trick, and perform tanh normalization
        # This library might be helpful: torch.distributions
        ############################
        # YOUR IMPLEMENTATION HERE #
        normal = Normal(mean, log_std.exp())
        action = normal.rsample()
        log_prob = normal.log_prob(action)
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        ############################
        return self._normalize(torch.tanh(action)), log_prob


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dims):
        super().__init__()
        self.fcs = mlp(num_states + num_actions, hidden_dims, 1)

    def forward(self, state, action):
        return self.fcs(torch.cat([state, action], dim=1)).squeeze()
