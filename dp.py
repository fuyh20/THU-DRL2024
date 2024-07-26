# coding: utf-8
import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register


register(id='SlipperyFrozenLake-v1',
    entry_point='gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': True})
register(id='FrozenLake-v1',
    entry_point='gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False})


"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [0, nS - 1] and actions in [0, nA - 1], P[state][action] is a list of
		tuple of the form (p_trans, next_state, reward, terminal) where
			- p_trans: float
				the transition probability of transitioning from "state" to "next_state" with "action"
			- next_state: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"next_state" with "action"
			- terminal: bool
			  True when "next_state" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, eps=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
      defined at beginning of file
    policy: np.array[nS]
      The policy to evaluate. Maps states to actions.
    eps: float
      Terminate policy evaluation when
        max |value_function(s) - next_value_function(s)| < eps
    Returns
    -------
    value_function: np.ndarray[nS]
      The value function of the given policy, where value_function[s] is
      the value of state s
    """

    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            value_function[s] = sum([p * (r + gamma * value_function[s_prime]) if not terminal else p * r for p, s_prime, r, terminal in P[s][policy[s]]])
            delta = max(delta, abs(v - value_function[s]))
        if delta < eps:
            break

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy, improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
      defined at beginning of file
    value_from_policy: np.ndarray
      The value calculated from evaluating the policy
    policy: np.array
      The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """

    new_policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(nS):
        action_value = np.zeros(nA)
        for a in range(nA):
            action_value[a] = sum([p * (r + gamma * value_from_policy[s_prime]) if not terminal else p * r for p, s_prime, r, terminal in P[s][a]])
        new_policy[s] = np.argmax(action_value)
            

    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, eps=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
      defined at beginning of file
    eps: float
      eps parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    while True:
        old_policy = np.copy(policy)

        value_function = policy_evaluation(P, nS, nA, policy, gamma=gamma, eps=eps)
        policy = policy_improvement(P, nS, nA, value_function, policy, gamma=gamma)

        if np.all(old_policy == policy):
            break

    ############################
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, eps=1e-3):
    """Learn value function and policy using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
      defined at beginning of file
    eps: float
      Terminate value iteration when
        max |value_function(s) - next_value_function(s)| < eps
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            action_value = np.zeros(nA)
            for a in range(nA):
                action_value[a] = sum([p * (r + gamma * value_function[s_prime]) if not terminal else p * r for p, s_prime, r, terminal in P[s][a]])
            policy[s] = np.argmax(action_value)
            value_function[s] = action_value[policy[s]]
            delta = max(delta, abs(v - value_function[s]))
        if delta < eps:
            break

    ############################
    return value_function, policy


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


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments
    # human render mode for the animation
    env = gym.make("FrozenLake-v1", render_mode="human")
    # env = gym.make("SlipperyFrozenLake-v1", render_mode="human")

    env = TimeLimit(env, max_episode_steps=100)
    P = env.P
    nS, nA = env.observation_space.n, env.action_space.n

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(P, nS, nA, gamma=0.9, eps=1e-3)
    render_single(env, p_pi, 100)
    print(p_pi)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(P, nS, nA, gamma=0.9, eps=1e-3)
    render_single(env, p_vi, 100)
    print(p_vi)
