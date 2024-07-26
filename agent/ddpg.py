import os
import torch
from copy import deepcopy
from models import Actor, Critic
from utils import get_schedule

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
        self.eps_schedule = get_schedule(eps_schedule)

    def __repr__(self):
        return "DDPGAgent"

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q = self.critic_net(state, action)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            Q_target = reward + (1 - done) * self.gamma * self.critic_target(next_state, next_action)
        
        critic_loss = torch.mean((Q - Q_target)**2 * (weights if weights is not None else 1))
        td_error = torch.abs(Q - Q_target).detach()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item(), td_error.mean().item()

    def update_actor(self, state):
        pred_action = self.actor_net(state)
        actor_loss = -self.critic_net(state, pred_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    @torch.no_grad()
    def get_action(self, state, sample=False):
        action = self.actor_net(torch.as_tensor(state).to(self.device))
        if sample:
            action += torch.randn_like(action) * self.eps_schedule(self.train_step)
            action = action.clamp_(self.actor_net.action_space.low, self.actor_net.action_space.high)
        return action.cpu().numpy()

    def update(self, batch, weights=None):
        # TODO: reward scaling
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
        torch.save(self.critic_net.state_dict(), os.path.join('models', name_prefix + 'critic.pt'))
        torch.save(self.actor_net.state_dict(), os.path.join('models', name_prefix + 'actor.pt'))

    def load(self, name_prefix='best_'):
        self.critic_net.load_state_dict(torch.load(os.path.join('models', name_prefix + 'critic.pt')))
        self.actor_net.load_state_dict(torch.load(os.path.join('models', name_prefix + 'actor.pt')))
