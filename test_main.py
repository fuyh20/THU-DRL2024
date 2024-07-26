import json
import torch
import numpy as np
from agent.td3 import TD3Agent
from agent.sac import SACAgent
from agent.ddpg import DDPGAgent
from omegaconf import OmegaConf
from dotmap import DotMap
import gymnasium as gym
from hydra.utils import instantiate
from copy import deepcopy
import utils
from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, \
    NormalizeObservation, TransformObservation, NormalizeReward, TransformReward

test_cases = 4
env = gym.make('LunarLanderContinuous-v2')
make_env = lambda: TransformReward(
        NormalizeReward(
            TransformObservation(
                NormalizeObservation(
                    ClipAction(
                        RecordEpisodeStatistics(
                            gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")
                        )
                    )
                ), lambda obs: np.clip(obs, -10, 10)
            )
        ), lambda reward: np.clip(reward, -10, 10)
    )
env = make_env()

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

def test_get_policy_loss():
    cfg = DotMap(OmegaConf.load('test/cfgs/ppo.yaml'))
    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)
    agent = instantiate(cfg, state_size=state_size, action_size=action_size, action_space=env.action_space, device='cpu')
    

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_policy_loss/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data = test_json['in']
        log_prob, old_log_prob, entropy = torch.tensor(input_data['log_prob']), torch.tensor(input_data['old_log_prob']), torch.tensor(input_data['advantage'])
        policy_loss = agent.get_policy_loss(log_prob, old_log_prob, entropy)
        assert torch.allclose(policy_loss, torch.tensor(test_json['out']), atol=1e-4), f'Policy loss does not match for {test_filename}'

    print(f'test_get_policy_loss passed! [10 pts]\n')



def test_get_value_loss():
    cfg = DotMap(OmegaConf.load('test/cfgs/ppo.yaml'))
    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)
    agent = instantiate(cfg, state_size=state_size, action_size=action_size, action_space=env.action_space, device='cpu')
    

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_value_loss/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data = test_json['in']
        value, old_value, returns = torch.tensor(input_data['value']), torch.tensor(input_data['old_value']), torch.tensor(input_data['returns'])
        value_loss = agent.get_value_loss(value, old_value, returns)
        assert torch.allclose(value_loss, torch.tensor(test_json['out']), atol=1e-4), f'Value loss does not match for {test_filename}'

    print(f'test_get_value_loss passed! [10 pts]\n')

def test_get_entropy_loss():
    cfg = DotMap(OmegaConf.load('test/cfgs/ppo.yaml'))
    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)
    agent = instantiate(cfg, state_size=state_size, action_size=action_size, action_space=env.action_space, device='cpu')
    

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_entropy_loss/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)

        entropy = torch.tensor(test_json['in'])
        entropy_loss = agent.get_entropy_loss(entropy)
        assert torch.allclose(entropy_loss, torch.tensor(test_json['out']), atol=1e-4), f'Entropy loss does not match for {test_filename}'

    print(f'test_get_entropy_loss passed! [10 pts]\n')

if __name__ == '__main__':
    results = []
    from tabulate import tabulate, SEPARATING_LINE
    results.append(run_test(test_get_policy_loss, points='10 pts'))
    results.append(run_test(test_get_value_loss, points='10 pts'))
    results.append(run_test(test_get_entropy_loss, points='10 pts'))
    results.append([SEPARATING_LINE, SEPARATING_LINE, SEPARATING_LINE])
    total_score = sum([int(r[-1].split(' ')[0]) for r in results if r[1] == 'Passed'])
    passed = sum([1 for r in results if r[1] == 'Passed'])
    results.append(['Total', f'{passed}/3', f'{total_score} pts'])

    print(tabulate(results, headers=["Test Name", "Result", "Score"]))