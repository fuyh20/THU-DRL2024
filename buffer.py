import torch
import random
import numpy as np
from collections import deque


def get_buffer(cfg, **args):
    assert type(cfg.nstep) == int and cfg.nstep > 0, 'nstep must be a positive integer'
    if not cfg.use_per:
        if cfg.nstep == 1:
            return ReplayBuffer(cfg.capacity, **args)
        else:
            return NStepReplayBuffer(cfg.capacity, cfg.nstep, cfg.gamma, **args)
    else:
        if cfg.nstep == 1:
            return PrioritizedReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, **args)
        else:
            return PrioritizedNStepReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, cfg.nstep, cfg.gamma, **args)


class ReplayBuffer:
    def __init__(self, capacity, state_size, seed, device):
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.actions = torch.zeros(capacity, dtype=torch.long).contiguous().pin_memory()
        self.rewards = torch.zeros(capacity, dtype=torch.float).contiguous().pin_memory()
        self.next_states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.dones = torch.zeros(capacity, dtype=torch.int).contiguous().pin_memory()

        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.states[self.idx] = torch.as_tensor(state)
        self.actions[self.idx] = torch.as_tensor(action)
        self.rewards[self.idx] = torch.as_tensor(reward)
        self.next_states[self.idx] = torch.as_tensor(next_state)
        self.dones[self.idx] = torch.as_tensor(done)

        # update counters
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        # using np.random.default_rng().choice is faster https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        batch = (
            self.states[sample_idxs].to(self.device, non_blocking=True),
            self.actions[sample_idxs].to(self.device, non_blocking=True),
            self.rewards[sample_idxs].to(self.device, non_blocking=True),
            self.next_states[sample_idxs].to(self.device, non_blocking=True),
            self.dones[sample_idxs].to(self.device, non_blocking=True)
        )
        return batch


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, seed, device):
        super().__init__(capacity, state_size, seed, device=device)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        """Get n-step state, action, reward and done for the transition, discard those rewards after done=True"""
        ############################
        # YOUR IMPLEMENTATION HERE #
        state, action, _, _ = self.n_step_buffer[0]
        done = False
        reward = 0
        for i in range(len(self.n_step_buffer)):
            _, _, _, done = self.n_step_buffer[i]
            if done:
                break
            reward += self.n_step_buffer[i][2] * (self.gamma ** i)
        ############################
        return state, action, reward, done

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, seed, device):
        # self.tree = SumTreeArray(capacity, dtype='float32')
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, seed, device=device)

    def add(self, transition):
        self.priorities[self.idx] = self.max_priority
        super().add(transition)

    def sample(self, batch_size):
        # sample_idxs = self.tree.sample(batch_size)
        sample_idxs = self.rng.choice(self.capacity, batch_size, p=self.priorities / self.priorities.sum(), replace=True)
        
        # Get the importance sampling weights for the sampled batch using the prioity values
        # For stability reasons, we always normalize weights by max(w_i) so that they only scale the
        # update downwards, whenever importance sampling is used, all weights w_i were scaled so that max_i w_i = 1.
        
        ############################
        # YOUR IMPLEMENTATION HERE #
        weights = np.power(self.size * self.priorities[sample_idxs], -self.beta)
        weights /= weights.max()

        weights = torch.as_tensor(weights).to(self.device)
        ############################
    
        batch = (
            self.states[sample_idxs].to(self.device, non_blocking=True),
            self.actions[sample_idxs].to(self.device, non_blocking=True),
            self.rewards[sample_idxs].to(self.device, non_blocking=True),
            self.next_states[sample_idxs].to(self.device, non_blocking=True),
            self.dones[sample_idxs].to(self.device, non_blocking=True)
        )
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha

        self.priorities[data_idxs] = priorities
        self.max_priority = max(self.priorities)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'


# Avoid Diamond Inheritance
class PrioritizedNStepReplayBuffer(PrioritizedReplayBuffer):
    # Implement the PrioritizedNStepReplayBuffer class if you want to, this is OPTIONAL
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, seed, device):
        ############################
        # YOUR IMPLEMENTATION HERE #
        super().__init__(capacity, eps, alpha, beta, state_size, seed, device=device)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma
        ############################
    def __repr__(self) -> str:
        return f'Prioritized{self.n_step}StepReplayBuffer'

    def add(self, transition):
        ############################
        # YOUR IMPLEMENTATION HERE #
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))
        ############################

    # def the other necessary class methods as your need
    def n_step_handler(self):
        """Get n-step state, action, reward and done for the transition, discard those rewards after done=True"""
        ############################
        # YOUR IMPLEMENTATION HERE #
        state, action, _, _ = self.n_step_buffer[0]
        done = False
        reward = 0
        for i in range(len(self.n_step_buffer)):
            _, _, _, done = self.n_step_buffer[i]
            if done:
                break
            reward += self.n_step_buffer[i][2] * (self.gamma ** i)
        ############################
        return state, action, reward, done
