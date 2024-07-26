import random
import logging
import numpy as np
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from copy import deepcopy
from utils import merge_videos, visualize, get_epsilon
from gymnasium.wrappers import RecordVideo
logger = logging.getLogger(__name__)


def eval(env, agent, episodes, seed):
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=episode + seed)
        done, truncated = False, False

        while not (done or truncated):
            state, _, done, truncated, info = env.step(agent.get_action(state).item())
        returns.append(info['episode']['r'].item())
    return np.mean(returns), np.std(returns)


def train(cfg, env, agent, buffer, seed, log_dict):
    eval_env = deepcopy(env)
    for key in log_dict.keys():
        log_dict[key].append([])
    
    done, truncated, best_reward = False, False, -np.inf
    state, _ = env.reset(seed=seed)
    for step in range(1, cfg.timesteps + 1):
        if done or truncated:
            state, _ = env.reset()
            done, truncated = False, False
            log_dict['train_returns'][-1].append(info['episode']['r'].item())
            log_dict['train_steps'][-1].append(step - 1)

        eps = get_epsilon(step, cfg.eps_min, cfg.eps_max, cfg.eps_steps, cfg.warmup_steps)

        if step < cfg.warmup_steps or random.random() < eps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state).item()

        next_state, reward, done, truncated, info = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))
        state = next_state

        if step > cfg.batch_size + cfg.nstep:
            if isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(cfg.batch_size)
                loss, td_error, Q = agent.update(batch, step, weights=weights)

                buffer.update_priorities(tree_idxs, td_error.cpu().numpy())
            elif isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(cfg.batch_size)
                loss, td_error, Q = agent.update(batch, step)
            else:
                raise RuntimeError("Unknown buffer")

            log_dict['Qs'][-1].append(Q)
            log_dict['losses'][-1].append(loss)

        if step % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
            log_dict['eval_steps'][-1].append(step - 1)
            log_dict['eval_returns'][-1].append(eval_mean)
            logger.info(f"Seed: {seed}, Step: {step}, Eval mean: {eval_mean}, Eval std: {eval_std}")
            if eval_mean > best_reward:
                best_reward = eval_mean
                agent.save(f'best_model_seed_{seed}.pt')

        if step % cfg.plot_interval == 0:
            visualize(step, f'{agent} with {buffer}', log_dict)

    agent.save(f'final_model_seed_{seed}.pt')
    visualize(step, f'{agent} with {buffer}', log_dict)

    env = RecordVideo(eval_env, f'final_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)

    agent.load(name=f'best_model_seed_{seed}.pt')  # use best model for visualization
    env = RecordVideo(eval_env, f'best_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
    merge_videos(f'final_videos_seed_{seed}')
    merge_videos(f'best_videos_seed_{seed}')
    env.close()
    return eval_mean
