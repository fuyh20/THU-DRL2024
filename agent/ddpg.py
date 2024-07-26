import os
import torch
import numpy as np
from torch import Tensor
from copy import deepcopy
from models import Actor, Critic
from utils import get_schedule

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

class DDPGAgent:
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep, target_update_interval, eps_schedule, device):
  
        self.critic_net = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic_net).to(device)
        self.actor_net = Actor(state_size, action_size, deepcopy(action_space), hidden_dim).to(device)
        self.actor_target = deepcopy(self.actor_net).to(device)

        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=lr_critic)
    
        self.tau = tau
        self.gamma = gamma ** nstep
        self.device = device
        self.target_update_interval = target_update_interval

        self.train_step = 0
        self.epsilon_schedule = get_schedule(eps_schedule)

    def __repr__(self):
        return "DDPGAgent"

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def eval(self):
        self.actor_net.eval()

    def train(self):
        self.actor_net.train()
    
    @jaxtyped(typechecker=beartype)
    # jaxtyped is a decorator that allows you to typecheck the shape of your input & output using beartype, please refer to https://github.com/patrick-kidger/jaxtyping/blob/main/docs/api/array.md for more information.
    def get_Qs(self, 
            state: Float[Tensor, "batch_size state_dim"], 
            action: Float[Tensor, "batch_size action_dim"], 
            reward: Float[Tensor, "batch_size"], 
            next_state: Float[Tensor, "batch_size state_dim"], 
            done: Int[Tensor, "batch_size"]
        ) -> tuple[Float[Tensor, "batch_size"], Float[Tensor, "batch_size"]]:
        """
        Obtain the Q and target Q values from the agent's Q networks.
        Hint: this is the get_Q and get_Q_target method of Homework 2 combined.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        Q = self.critic_net(state, action)
        Q_target = reward + self.gamma * (1 - done) * self.critic_target(next_state, self.actor_target(next_state))
        ############################
        return Q, Q_target
    
    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q, Q_target = self.get_Qs(state, action, reward, next_state, done)
        
        critic_loss = torch.mean((Q - Q_target)**2 * (weights if weights is not None else 1))
        td_error = torch.abs(Q - Q_target).detach()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item(), td_error.mean().item()

    @jaxtyped(typechecker=beartype)
    def get_actor_loss(self, 
            state: Float[Tensor, "batch_size state_dim"]
        ) -> Float[Tensor, ""]:
        """
        Obtain actor loss given state using the agent's Q and policy networks.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        actor_loss = -self.critic_net(state, self.actor_net(state)).mean()
        ############################
        return actor_loss
    
    def update_actor(self, state):
        actor_loss = self.get_actor_loss(state)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def get_action(self, 
            state: Float[np.ndarray, "state_dim"], 
            sample: bool = False
        ) -> Float[np.ndarray, "action_dim"]:
        """
        Use the policy network to obtain an action given the state.
        If sample, add noise to the action. The magnitude of the noise is determined by current epsilon.
        Hint: if you don't know what epsilon is, try looking for it in __init__
        Remember to clamp the action to the action_space's low and high values.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor_net(state)
        if sample:
            action += torch.randn_like(action) * self.epsilon_schedule(self.train_step)
        action = torch.clamp(action, self.actor_net.action_space.low, self.actor_net.action_space.high)
        action = action.cpu().numpy()
        ############################
        return action
    
    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch

        critic_loss, td_error = self.update_critic(state, action, reward, next_state, done, weights)
        actor_loss = self.update_actor(state)

        # update the target_qnet
        if not self.train_step % self.target_update_interval:
            self.soft_update(self.critic_target, self.critic_net)
            self.soft_update(self.actor_target, self.actor_net)

        self.train_step += 1
        return {'critic_loss': critic_loss, 'actor_loss': actor_loss, 'td_error': td_error}
    
    def save(self, name_prefix='best_'):
        os.makedirs('models', exist_ok=True)
        torch.save(self.critic_net.state_dict(), os.path.join('models', name_prefix + '_critic.pt'))
        torch.save(self.actor_net.state_dict(), os.path.join('models', name_prefix + '_actor.pt'))

    def load(self, name_prefix='best_'):
        self.critic_net.load_state_dict(torch.load(os.path.join('models', name_prefix + '_critic.pt')))
        self.actor_net.load_state_dict(torch.load(os.path.join('models', name_prefix + '_actor.pt')))
