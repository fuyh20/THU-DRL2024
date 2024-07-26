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

test_cases = 4
env = gym.make('LunarLanderContinuous-v2')

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

def test_get_Qs_ddpg():
    cfg = DotMap(OmegaConf.load('test/cfgs/ddpg.yaml'))
    agent = instantiate(cfg, state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], action_space=deepcopy(env.action_space), device='cpu')
    
    agent.actor_net.load_state_dict(torch.load('test/ckpts/ddpg_actor.pt'))
    agent.actor_target.load_state_dict(torch.load('test/ckpts/ddpg_actor.pt'))
    agent.critic_net.load_state_dict(torch.load('test/ckpts/ddpg_critic.pt'))
    agent.critic_target.load_state_dict(torch.load('test/ckpts/ddpg_critic_target.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_Qs_ddpg/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data = test_json['in']
        state, action, reward, next_state, done = torch.tensor(input_data['state']), torch.tensor(input_data['action']), torch.tensor(input_data['reward']), torch.tensor(input_data['next_state']), torch.tensor(input_data['done'])
        Q, Q_target = agent.get_Qs(state, action, reward, next_state, done)
        assert torch.allclose(Q, torch.tensor(test_json['out']['Q']), atol=1e-4), f'Q value does not match for {test_filename}'
        assert torch.allclose(Q_target, torch.tensor(test_json['out']['Q_target']), atol=1e-4), f'Q_target value does not match for {test_filename}'

    print(f'test_get_Qs_ddpg passed! [10 pts]\n')



def test_get_actor_loss_ddpg():
    cfg = DotMap(OmegaConf.load('test/cfgs/ddpg.yaml'))
    agent = instantiate(cfg, state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], action_space=deepcopy(env.action_space), device='cpu')
    
    agent.actor_net.load_state_dict(torch.load('test/ckpts/ddpg_actor.pt'))
    agent.critic_net.load_state_dict(torch.load('test/ckpts/ddpg_critic.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_actor_loss_ddpg/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        state = torch.tensor(test_json['in'])
        loss = agent.get_actor_loss(state)
        assert torch.allclose(loss, torch.tensor(test_json['out']), atol=1e-4), f'Actor loss does not match for {test_filename}'

    print(f'test_get_actor_loss_ddpg passed! [5 pts]\n')

def test_get_Qs_td3():
    cfg = DotMap(OmegaConf.load('test/cfgs/td3.yaml'))
    agent = instantiate(cfg, state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], action_space=deepcopy(env.action_space), device='cpu')
    agent.policy_noise = 0 # Disable noise for testing

    agent.actor_net.load_state_dict(torch.load('test/ckpts/td3_actor.pt'))
    agent.actor_target.load_state_dict(torch.load('test/ckpts/td3_actor_target.pt'))
    agent.critic_net.load_state_dict(torch.load('test/ckpts/td3_critic.pt'))
    agent.critic_target.load_state_dict(torch.load('test/ckpts/td3_critic_target.pt'))
    agent.critic_net_2.load_state_dict(torch.load('test/ckpts/td3_critic_2.pt'))
    agent.critic_target_2.load_state_dict(torch.load('test/ckpts/td3_critic_target_2.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_Qs_td3/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        input_data = test_json['in']
        state, action, reward, next_state, done = torch.tensor(input_data['state']), torch.tensor(input_data['action']), torch.tensor(input_data['reward']), torch.tensor(input_data['next_state']), torch.tensor(input_data['done'])
        Q, Q2, Q_target = agent.get_Qs(state, action, reward, next_state, done)
        assert torch.allclose(Q, torch.tensor(test_json['out']['Q']), atol=1e-4), f'Q value does not match for {test_filename}'
        assert torch.allclose(Q2, torch.tensor(test_json['out']['Q2']), atol=1e-4), f'Q2 value does not match for {test_filename}'
        assert torch.allclose(Q_target, torch.tensor(test_json['out']['Q_target']), atol=1e-4), f'Q_target value does not match for {test_filename}'

    print(f'test_get_Qs_td3 passed! [10 pts]\n')

def test_get_alpha_loss_sac():
    cfg = DotMap(OmegaConf.load('test/cfgs/sac.yaml'))
    agent = instantiate(cfg, state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], action_space=deepcopy(env.action_space), device='cpu')
    

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/get_alpha_loss_sac/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        action_log_prob = torch.tensor(test_json['in'])
        loss = agent.get_alpha_loss(action_log_prob)
        assert torch.allclose(loss, torch.tensor(test_json['out']), atol=1e-4), f'Alpha loss does not match for {test_filename}'

    print(f'test_get_alpha_loss_sac passed! [5 pts]\n')

def test_forward_sac():
    cfg = DotMap(OmegaConf.load('test/cfgs/sac.yaml'))
    agent = instantiate(cfg, state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], action_space=deepcopy(env.action_space), device='cpu')
    
    agent.actor_net.load_state_dict(torch.load('test/ckpts/sac_actor.pt'))

    for k in range(1, test_cases + 1):
        test_filename = f'test/test_cases/forward_sac/{k}.json'
        print(f'Testing {test_filename}...')
        
        test_json = load_json(test_filename)
        state = torch.tensor(test_json['in'])
        mean, std = agent.actor_net(state)
        for key in ['mean', 'std']:
            assert torch.allclose(eval(key), torch.tensor(test_json['out'][key]), atol=1e-4), f'{key} does not match for {test_filename}'

    print(f'test_forward_sac passed! [10 pts]\n')

if __name__ == '__main__':
    results = []
    from tabulate import tabulate, SEPARATING_LINE
    results.append(run_test(test_get_Qs_ddpg, points='10 pts'))
    results.append(run_test(test_get_actor_loss_ddpg, points='5 pts'))
    results.append(run_test(test_get_Qs_td3, points='5 pts'))
    results.append(run_test(test_get_alpha_loss_sac, points='5 pts'))
    results.append(run_test(test_forward_sac, points='5 pts'))
    results.append([SEPARATING_LINE, SEPARATING_LINE, SEPARATING_LINE])
    total_score = sum([int(r[-1].split(' ')[0]) for r in results if r[1] == 'Passed'])
    passed = sum([1 for r in results if r[1] == 'Passed'])
    results.append(['Total', f'{passed}/5', f'{total_score} pts'])

    print(tabulate(results, headers=["Test Name", "Result", "Score"]))