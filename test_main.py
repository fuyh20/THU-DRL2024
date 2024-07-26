import json
import numpy as np
from dp import value_iteration, policy_iteration
from td import epsilon_greedy_policy, Q_learning_step, Sarsa_step

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def test_value_iteration():
    m = 4
    for k in range(1, m + 1):
        input_filename = f'test/test_cases/dp/value_iteration/{k}.in'
        output_filename = f'test/test_cases/dp/value_iteration/{k}.out'
        print(f'Testing {input_filename}...')
        
        input_data = load_json(input_filename)
        P_json, nS, nA = input_data['P'], input_data['nS'], input_data['nA']
        
        nP = max([len(P_json[s][a]) for s in P_json for a in P_json[s]])
        P = [[[[0 for _ in range(4)] for _ in range(nP)] for _ in range(nA)] for _ in range(nS)]

        for s in P_json:
            for a in P_json[s]:
                for i, transition in enumerate(P_json[s][a]):
                    terminal = True if transition[3] else False
                    P[int(s)][int(a)][i] = [transition[0], transition[1], transition[2], terminal]
        
        gamma, eps = input_data['gamma'], input_data['eps']
        
        value_function, policy = value_iteration(P, nS, nA, gamma, eps)
        
        expected_output = load_json(output_filename)
        expected_value_function = np.array(expected_output['value_function'])
        expected_policy = np.array(expected_output['policy'])
        
        assert np.allclose(value_function, expected_value_function, atol=0.005), f'Value function does not match for {input_filename}'
       
        print(f'Test {k} passed!')

def test_policy_iteration():
    m = 4
    for k in range(1, m + 1):
        input_filename = f'test/test_cases/dp/policy_iteration/{k}.in'
        output_filename = f'test/test_cases/dp/policy_iteration/{k}.out'
        print(f'Testing {input_filename}...')
        
        input_data = load_json(input_filename)
        P_json, nS, nA = input_data['P'], input_data['nS'], input_data['nA']
        
        nP = max([len(P_json[s][a]) for s in P_json for a in P_json[s]])
        P = [[[[0 for _ in range(4)] for _ in range(nP)] for _ in range(nA)] for _ in range(nS)]

        for s in P_json:
            for a in P_json[s]:
                for i, transition in enumerate(P_json[s][a]):
                    terminal = True if transition[3] else False
                    P[int(s)][int(a)][i] = [transition[0], transition[1], transition[2], terminal]
        
        gamma, eps = input_data['gamma'], input_data['eps']
        
        value_function, policy = policy_iteration(P, nS, nA, gamma, eps)
        
        expected_output = load_json(output_filename)
        expected_value_function = np.array(expected_output['value_function'])
        expected_policy = np.array(expected_output['policy'])
        
        assert np.allclose(value_function, expected_value_function, atol=0.005), f'Value function does not match for {input_filename}'
       
        print(f'Test {k} passed!')


def test_epsilon_greedy_policy():
    m = 10
    for k in range(1, m + 1):
        input_filename = f'test/test_cases/td/epsilon_greedy_policy/{k}.in'
        output_filename = f'test/test_cases/td/epsilon_greedy_policy/{k}.out'
        print(f'Testing {input_filename}...')

        input_data = load_json(input_filename)
        nS, nA = input_data['nS'], input_data['nA']
        Q_function = np.array(input_data['Q_function'])
        eps = input_data['eps']
        
        policy = epsilon_greedy_policy(nS, nA, Q_function, eps)
        
        expected_output = load_json(output_filename)
        expected_policy = np.array(expected_output['policy'])
        
        assert np.allclose(policy, expected_policy, atol=0.005), f'Policy does not match for {input_filename}'
        
        print(f'Test {k} passed!')
        
def test_Q_learning():
    m = 10
    for k in range(1, m + 1):
        input_filename = f'test/test_cases/td/Q_learning/{k}.in'
        output_filename = f'test/test_cases/td/Q_learning/{k}.out'
        print(f'Testing {input_filename}...')

        input_data = load_json(input_filename)
        Q_function = np.array(input_data['Q_function'])
        state = input_data['state']
        action = input_data['action']
        next_state = input_data['next_state']
        next_action = input_data['next_action']
        reward = input_data['reward']
        terminal = input_data['terminal']
        alpha = input_data['alpha']
        gamma = input_data['gamma']
        next_Q_function = Q_learning_step(Q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma)
        expected_output = load_json(output_filename)
        expected_next_Q_function = np.array(expected_output['next_Q_function'])
        
        assert np.allclose(next_Q_function, expected_next_Q_function, atol=0.005), f'Q function does not match for {input_filename}'
        
        print(f'Test {k} passed!')
        
def test_Sarsa():
    m = 10
    for k in range(1, m + 1):
        input_filename = f'test/test_cases/td/Sarsa/{k}.in'
        output_filename = f'test/test_cases/td/Sarsa/{k}.out'
        print(f'Testing {input_filename}...')

        input_data = load_json(input_filename)
        Q_function = np.array(input_data['Q_function'])
        state = input_data['state']
        action = input_data['action']
        next_state = input_data['next_state']
        next_action = input_data['next_action']
        reward = input_data['reward']
        terminal = input_data['terminal']
        alpha = input_data['alpha']
        gamma = input_data['gamma']
        next_Q_function = Sarsa_step(Q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma)
        expected_output = load_json(output_filename)
        expected_next_Q_function = np.array(expected_output['next_Q_function'])
        
        assert np.allclose(next_Q_function, expected_next_Q_function, atol=0.005), f'Q function does not match for {input_filename}'
        
        print(f'Test {k} passed!')
        
