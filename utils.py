import os
import glob
import torch
import sys
import shutil
import random
import logging
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from moviepy.editor import VideoFileClip, concatenate_videoclips


def config_logging(log_file="main.log"):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(asctime)s: [%(levelname)s]: %(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Set up the FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])


def visualize(step, title, log_dict):
    train_window, loss_window, q_window = 10, 100, 100
    plt.figure(figsize=(20, 6))

    # plot train and eval returns
    plt.subplot(1, 3, 1)
    plt.title('frame %s. score: %s' % (step, np.mean(log_dict['train_returns'][-1][-train_window:])))
    plot_scores(log_dict['train_returns'], log_dict['train_steps'], train_window, label='train')
    if min([len(log_dict['eval_steps'][i]) for i in range(len(log_dict['eval_steps']))]) > 0:
        plot_scores(log_dict['eval_returns'], log_dict['eval_steps'], window=1, label='eval')
    plt.legend()
    plt.xlabel('step')

    # plot td losses
    plt.subplot(1, 3, 2)
    plt.title('loss')
    plot_scores(log_dict['losses'], window=loss_window, label='loss')
    plt.xlabel('step')
    plt.subplot(1, 3, 3)

    # plot q values
    plt.title('q_values')
    plot_scores(log_dict['Qs'], window=q_window, label='q_values')
    plt.xlabel('step')
    plt.suptitle(title, fontsize=16)
    plt.savefig('results.png')
    plt.close()


def moving_average(a, n):
    if len(a) <= n:
        return a
    ret = np.cumsum(a, dtype=float, axis=-1)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n).tolist()

def get_epsilon(step, eps_min, eps_max, eps_steps, warmup_steps):
    """
    Return the linearly descending epsilon of the current step for the epsilon-greedy policy. 
    The value of epsilon will keep at eps_max before warmup_steps, and after eps_steps, it will keep at eps_min.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    if step < warmup_steps:
        return eps_max
    elif step >= eps_steps:
        return eps_min
    else:
        return eps_max - (eps_max - eps_min) * (step - warmup_steps) / (eps_steps - warmup_steps)
    ############################

def pad_and_get_mask(lists):
    """
    Pad a list of lists with zeros and return a mask of the same shape.
    """
    lens = [len(l) for l in lists]
    max_len = max(lens)
    arr = np.zeros((len(lists), max_len), int)
    mask = np.arange(max_len) < np.array(lens)[:, None]
    arr[mask] = np.concatenate(lists)
    return np.ma.array(arr, mask=~mask)

# interpolate for different length lists


def plot_scores(scores, steps=None, window=100, label=None):
    avg_scores = [moving_average(score, window) for score in scores]
    if steps is not None:
        for i in range(len(scores)):
            avg_scores[i] = np.interp(np.arange(steps[i][-1]), [0] + steps[i][window - 1:], [0.0] + avg_scores[i])
    if len(scores) > 1:
        avg_scores = pad_and_get_mask(avg_scores)
        scores = avg_scores.mean(axis=0)
        scores_l = avg_scores.mean(axis=0) - avg_scores.std(axis=0)
        scores_h = avg_scores.mean(axis=0) + avg_scores.std(axis=0)
        idx = list(range(len(scores)))
        plt.fill_between(idx, scores_l, scores_h, where=scores_h > scores_l, interpolate=True, alpha=0.25)
    else:
        scores = avg_scores[0]
    plt.plot(scores, label=label)


def merge_videos(video_dir):
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    videos = sorted(videos, key=lambda x: int(x.split("-")[-1].split(".")[0]))
    clip = concatenate_videoclips([VideoFileClip(video) for video in videos])
    os.makedirs('videos', exist_ok=True)
    clip.write_videofile(os.path.join('videos', f"{video_dir}.mp4"))
    shutil.rmtree(video_dir)


def set_seed_everywhere(env: gym.Env, seed=0):
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_space_shape(space, is_vector_env=False):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, MultiDiscrete):
        return space.nvec[0]
    elif isinstance(space, Box):
        space_shape = space.shape[1:] if is_vector_env else space.shape
        if len(space_shape) == 1:
            return space_shape[0]
        else:
            return space_shape  # image observation
    else:
        raise ValueError(f"Space not supported: {space}")
