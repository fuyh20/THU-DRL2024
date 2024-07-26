import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit


"""
For Q_learning, Sarsa
the parameters nS, nA, gamma are defined as follows:

	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
    alpha: float
        Learning step rate for Q-learning and Sarsa
    state: int
        denotes the current state (in range [0, nS - 1])
    action: int
        denotes the action we take at current state (in range [0, nA - 1])
    next_state: int
        denotes the state we transition to (in range [0, nS - 1])
    next_action: int
        denotes the action we are going to take at the next state(in range [0, nA - 1])
    reward: int
        either 0 or 1, the reward for transitioning from "state" to "next_state" with "action"
    terminal: bool
        True when "next_state" is a terminal state (hole or goal), False otherwise
"""


def epsilon_greedy_policy(nS, nA, Q_function, eps=0.5):
    """Get the epsilon greedy policy from the current Q function.

    Parameters
    ----------
    nS, nA: defined at the beginning of the file
    Q_function: np.array[nS][nA]
        The current Q value for the given state and action
    eps: float
        The exploration factor epsilon
    Returns
    -------
    policy: np.array[nS][nA]
        An array of floats, policy[s][a] is the posibility of taking action a at state s
    """

    policy = np.zeros((nS, nA))

    ############################
    # YOUR IMPLEMENTATION HERE #
    
    action_star = np.argmax(Q_function, axis=1)
    idx = np.arange(nS)
    
    policy[idx, action_star] = 1 - eps + eps / nA
    policy[policy == 0] = eps / nA
    
    ############################

    return policy


def sample_action(policy, state):
    """Sample action to take at state s according to the current policy

    Parameters
    ----------
    policy: np.array[nS][nA]
        An array of floats, policy[s][a] is the possibility of taking action a at state s
    state: int
        Current state to take action
    Returns
    -------
    action: int
        The action to take at state s
    """

    action = 0

    ############################
    # YOUR IMPLEMENTATION HERE #

    nA = policy.shape[1]
    action = np.random.choice(nA, 1, p=policy[state]).item()

    ############################

    return action


def Q_learning_step(Q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma):
    """Update the Q function through Q learning algorithm.

    Parameters
    ----------
    state, action, reward, next_state, terminal, alpha: defined at the beginning of the file
    Q_function: np.array[nS][nA]
        The current Q value for the given state and action
    Returns
    -------
    next_Q_function: np.array[nS][nA]
        The updated Q value through one step Q learning.
    """

    next_Q_function = np.zeros(Q_function.shape)

    ############################
    # YOUR IMPLEMENTATION HERE #

    next_Q_function = Q_function
    next_Q_function[state][action] = \
        Q_function[state][action] + alpha * (reward + gamma * np.max(Q_function[next_state]) - Q_function[state][action]) \
        if not terminal else Q_function[state][action] + alpha * (reward - Q_function[state][action])

    ############################

    return next_Q_function


def Sarsa_step(Q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma):
    """Update the Q function through Sarsa algorithm.

    Parameters
    ----------
    state, action, reward, next_state, terminal, alpha: defined at the beginning of the file
    Q_function: np.array[nS][nA]
        The current Q value for the given state and action
    Returns
    -------
    next_Q_function: np.array[nS][nA]
        The updated Q value through one step Sarsa.
    """

    next_Q_function = np.zeros(Q_function.shape)

    ############################
    # YOUR IMPLEMENTATION HERE #

    next_Q_function = Q_function
    next_Q_function[state][action] = \
        Q_function[state][action] + alpha * (reward + gamma * Q_function[next_state][next_action] - Q_function[state][action]) \
        if not terminal else Q_function[state][action] + alpha * (reward - Q_function[state][action])

    ############################

    return next_Q_function


def learn(learning_step, episodes=5000, max_steps=100, alpha=0.8, gamma=0.9):
    """
    This function does not need to be modified.
    Perform Q learning or Sarsa learning and return the resulted Q function
    """
    # make training environment, render as ansi blocks
    env = gym.make('FrozenLake-v1', render_mode='ansi', is_slippery=False)
    # the environment will stop and return truncated=True in case it gets stuck
    env = TimeLimit(env, max_steps)

    nS, nA = env.observation_space.n, env.action_space.n
    Q_function = np.zeros((nS, nA))
    
    # annal the epsilon to estimate a GLIE policy
    eps_annaling = 1 / episodes

    # loop for training episodes
    for episode in range(episodes):
        state, _ = env.reset()
        terminal, truncated = False, False

        while True:
            policy = epsilon_greedy_policy(nS, nA, Q_function, eps=1 - episode * eps_annaling)
            action = sample_action(policy, state)

            next_state, reward, terminal, truncated, _ = env.step(action)  
            next_action = sample_action(policy, next_state)
            Q_function = learning_step(Q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma)
            
            state = next_state
        
            if terminal or truncated:
                break
    
    return Q_function, Q_function.argmax(axis=1)


def render_single(env, policy, max_steps=100):
    """This function does not need to be modified.
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    state, _ = env.reset()
    while True:
        env.render()
        time.sleep(0.25)
        action = policy[state]
        state, reward, terminal, truncated, _ = env.step(action)
        episode_reward += reward
        if terminal or truncated:
            break
    env.render()
    if not terminal:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# You may change the parameters in the functions below
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)

    print("\n" + "-" * 25 + "\nBeginning Q Learning\n" + "-" * 25)
    Q_function, p_q_learning = learn(Q_learning_step, alpha=0.8, gamma=0.9)
    render_single(env, p_q_learning)

    print("\n" + "-" * 25 + "\nBeginning Sarsa\n" + "-" * 25)
    Q_function, p_sarsa = learn(Sarsa_step, alpha=0.8, gamma=0.9)
    render_single(env, p_sarsa)
