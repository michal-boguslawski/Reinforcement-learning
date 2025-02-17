import gymnasium as gym
from gymnasium.wrappers import RecordVideo, Autoreset
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers.vector import DtypeObservation, NumpyToTorch, TransformReward
import torch as T
import os
import numpy as np


def change_reward_type(rew):
    return rew.to(T.float)


def make_env(env_id: str, num_envs: int = 2, dtype: T.dtype = np.float32, device: T.device = T.device('cpu'), env_type = AsyncVectorEnv, name_prefix = None, make_video: bool = False, **kwargs):
        
    envs_list = []
    for i in range(num_envs):
        env = gym.make(env_id, render_mode='rgb_array', **kwargs)
        env = Autoreset(env)
        # if i == 0:
        if make_video:
            env = RecordVideo(env, video_folder='.//save_videos', fps=100, name_prefix=name_prefix)
        envs_list.append(lambda: env)
    envs =  env_type(envs_list) 
    envs = DtypeObservation(envs, dtype=dtype)
    envs = NumpyToTorch(envs, device=device)
    envs = TransformReward(envs, func=change_reward_type)
    return envs

def play_game(env_id: str, hx_start: T.Tensor, agent: T.nn.Module, num_games: int, name_prefix: str | None = None, device: T.device = T.device('cpu'), make_video: bool = False):
    if make_video:
        if name_prefix is None:
            name_prefix = env_id
        os.makedirs('save_videos', exist_ok=True)
    env = make_env(env_id=env_id, num_envs=1, device=device, env_type=SyncVectorEnv, name_prefix=name_prefix, make_video=make_video)
    
    games = 0
    total_rewards = T.zeros(1)
    state, _ = env.reset()
    hx = hx_start.clone()
    while games < num_games:
        action, _, hx = agent.action(state, hx)
        used_action = T.tanh(action)
        state, reward, terminated, truncated, _ = env.step(used_action)
        total_rewards += reward.cpu()
        done = T.logical_or(terminated, truncated)
        if done:
            games += 1
            hx = hx_start.clone()
    env.close()
    return total_rewards / num_games