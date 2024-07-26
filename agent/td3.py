import torch
from torch import Tensor
from models import Critic
from copy import deepcopy
from agent.ddpg import DDPGAgent

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

class TD3Agent(DDPGAgent):
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep,
                 target_update_interval, noise_clip, policy_noise, policy_update_interval, eps_schedule, device):
        super().__init__(state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep, target_update_interval, eps_schedule, device)
        
        self.critic_net_2 = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target_2 = deepcopy(self.critic_net_2).to(device)
        self.critic_optimizer_2 = torch.optim.AdamW(self.critic_net_2.parameters(), lr=lr_critic)

        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_update_interval = policy_update_interval

    def __repr__(self):
        return "TD3Agent"

    @jaxtyped(typechecker=beartype)
    def get_Qs(self, 
            state: Float[Tensor, "batch_size state_dim"], 
            action: Float[Tensor, "batch_size action_dim"], 
            reward: Float[Tensor, "batch_size"], 
            next_state: Float[Tensor, "batch_size state_dim"], 
            done: Int[Tensor, "batch_size"]
        ) -> tuple[Float[Tensor, "batch_size"], Float[Tensor, "batch_size"], Float[Tensor, "batch_size"]]:
        """
        Obtain the two Q value estimates and the target Q value from the twin Q networks.
        Hint: remember to use target policy smoothing.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        Q = self.critic_net(state, action)
        Q2 = self.critic_net_2(state, action)
        with torch.no_grad():
            action_target = self.actor_target(next_state)
            noise = torch.clamp(torch.randn_like(action_target) * self.policy_noise, -self.noise_clip, self.noise_clip)
            action_target = torch.clamp(action_target + noise, self.actor_net.action_space.low, self.actor_net.action_space.high)
            Q_target = reward + self.gamma * (1 - done) * torch.min(self.critic_target(next_state, action_target), self.critic_target_2(next_state, action_target))
        ############################
        return Q, Q2, Q_target

    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q, Q2, Q_target = self.get_Qs(state, action, reward, next_state, done)
        with torch.no_grad():
            td_error = torch.abs(Q - Q_target)
    
        if weights is None:
            critic_loss = torch.mean((Q - Q_target)**2)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2)
        else:
            critic_loss = torch.mean((Q - Q_target)**2 * weights)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2 * weights)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()
        return critic_loss.item(), critic_loss_2.item(), td_error.mean().item()
    
    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch
        critic_loss, critic_loss_2, td_error = self.update_critic(state, action, reward, next_state, done, weights)

        log_dict = {'critic_loss': critic_loss, 'critic_loss_2': critic_loss_2, 'td_error': td_error}
        
        # perform delayed policy updates every self.policy_update_interval step, and add actor_loss to log_dict
        ############################
        # YOUR IMPLEMENTATION HERE #
        if not self.train_step % self.policy_update_interval:
            actor_loss = self.update_actor(state)
            self.soft_update(self.actor_target, self.actor_net)
            log_dict['actor_loss'] = actor_loss
        ############################
        if not self.train_step % self.target_update_interval:
            self.soft_update(self.critic_target_2, self.critic_net_2)
            self.soft_update(self.critic_target, self.critic_net)

        self.train_step += 1
        return log_dict
