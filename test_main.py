import json
import torch
import numpy as np
from agent import DQNAgent
from omegaconf import OmegaConf
from dotmap import DotMap

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def run_test(test_function, points='5 pts'):
    test_name = test_function.__name__
    try:
        test_function()
        test_results = [test_name, 'Passed', points]
    except Exception as e:
        test_results = [test_name, 'Failed', points]
        print(f'{test_name} failed: {e}\n')
    
    return test_results

def test_get_Q():
    test_cases = 4
    cfg = DotMap(OmegaConf.load('test/config.yaml'))
    agent = DQNAgent(state_size=4, action_size=2, cfg=cfg.agent, device='cpu', compile=False)
    
    agent.q_net.load_state_dict(torch.load('test/q_net.pt'))
    agent.target_net.load_state_dict(torch.load('test/target_net.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_Q/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data, expected_output = test_json['in'], test_json['out']
        state, action = torch.tensor(input_data['state']), torch.tensor(input_data['action'])
        out = agent.get_Q(state, action).squeeze()
        expected_output = torch.tensor(expected_output)
        assert torch.allclose(out, expected_output), f'Q value does not match for {test_filename}'

    print(f'test_get_Q passed! [10 pts]\n')

def test_get_Q_target():
    test_cases = 4
    cfg = DotMap(OmegaConf.load('test/config.yaml'))
    agent = DQNAgent(state_size=4, action_size=2, cfg=cfg.agent, device='cpu', compile=False)
    
    agent.q_net.load_state_dict(torch.load('test/q_net.pt'))
    agent.target_net.load_state_dict(torch.load('test/target_net.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_Q_target/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data, expected_output = test_json['in'], test_json['out']
        reward, done, next_state = torch.tensor(input_data['reward']), torch.tensor(input_data['done']), torch.tensor(input_data['next_state'])
        out = agent.get_Q_target(reward, done, next_state)
        expected_output = torch.tensor(expected_output)
        
        assert torch.allclose(out, expected_output), f'Q target value does not match for {test_filename}'

    print(f'test_get_Q_target passed! [5 pts]\n')

def test_get_double_Q_target():
    test_cases = 4
    cfg = DotMap(OmegaConf.load('test/config.yaml'))
    cfg.agent.use_double=True
    agent = DQNAgent(state_size=4, action_size=2, cfg=cfg.agent, device='cpu', compile=False)
    
    agent.q_net.load_state_dict(torch.load('test/q_net.pt'))
    agent.target_net.load_state_dict(torch.load('test/target_net.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_double_Q_target/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data, expected_output = test_json['in'], test_json['out']
        reward, done, next_state = torch.tensor(input_data['reward']), torch.tensor(input_data['done']), torch.tensor(input_data['next_state'])
        out = agent.get_Q_target(reward, done, next_state)
        expected_output = torch.tensor(expected_output)
        
        assert torch.allclose(out, expected_output), 'Double Q target value does not match for {test_filename}'

    print(f'test_get_double_Q_target passed! [5 pts]\n')
     
def test_get_action():
    test_cases = 4
    cfg = DotMap(OmegaConf.load('test/config.yaml'))
    agent = DQNAgent(state_size=4, action_size=2, cfg=cfg.agent, device='cpu', compile=False)
    
    agent.q_net.load_state_dict(torch.load('test/q_net.pt'))
    agent.target_net.load_state_dict(torch.load('test/target_net.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_action/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data, expected_output = test_json['in'], test_json['out']
        state = np.array(input_data)
        out = agent.get_action(state)
        expected_output = np.array(expected_output)
        
        assert np.allclose(out, expected_output), f'Action does not match for {test_filename}'

    print(f'test_get_action passed! [5 pts]\n')

def test_dueling_forward():
    test_cases = 4
    cfg = DotMap(OmegaConf.load('test/config.yaml'))
    cfg.agent.use_dueling = True
    agent = DQNAgent(state_size=4, action_size=2, cfg=cfg.agent, device='cpu', compile=False)
    
    agent.q_net.load_state_dict(torch.load('test/dueling_q_net.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/dueling_Q_forward/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data, expected_output = test_json['in'], test_json['out']
        state = torch.tensor(input_data)
        out = agent.q_net(state)
        expected_output = torch.tensor(expected_output)
        
        assert torch.allclose(out, expected_output), f'Q value does not match for {test_filename}'

    print(f'test_dueling_forward passed! [5 pts]\n')

if __name__ == '__main__':
    results = []
    from tabulate import tabulate, SEPARATING_LINE
    results.append(run_test(test_get_Q, points='10 pts'))
    results.append(run_test(test_get_Q_target))
    results.append(run_test(test_get_double_Q_target))
    results.append(run_test(test_get_action))
    results.append(run_test(test_dueling_forward))
    results.append([SEPARATING_LINE, SEPARATING_LINE, SEPARATING_LINE])
    total_score = sum([int(r[-1].split(' ')[0]) for r in results if r[1] == 'Passed'])
    passed = sum([1 for r in results if r[1] == 'Passed'])
    results.append(['Total', f'{passed}/5', f'{total_score} pts'])

    print(tabulate(results, headers=["Test Name", "Result", "Score"]))