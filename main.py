import hydra
import utils
import torch
import logging
from agent import DQNAgent
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from core import train
from dotmap import DotMap
from buffer import get_buffer
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from omegaconf import OmegaConf
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    cfg = DotMap(OmegaConf.to_container(cfg, resolve=True))
    env = RecordEpisodeStatistics(gym.make(cfg.env_name, render_mode="rgb_array"))

    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)

    log_dict = {'losses':[], 'Qs': [], 'eval_steps': [], 'eval_returns': [], 'train_steps': [], 'train_returns': []}
    for seed in cfg.seeds:
        utils.set_seed_everywhere(env, seed)
        buffer = get_buffer(cfg.buffer, state_size=state_size, seed=seed, device=device)
        agent = DQNAgent(state_size=state_size, action_size=action_size, cfg=cfg.agent, device=device)
        logger.info(f"Training seed {seed} for {cfg.train.timesteps} timesteps with {agent} and {buffer}")
        eval_mean = train(cfg.train, env, agent, buffer, seed, log_dict)
        logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")


if __name__ == "__main__":
    main()
