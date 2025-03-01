import gymnasium as gym
from gymnasium.wrappers import RecordVideo, Autoreset, TimeLimit
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers.vector import DtypeObservation, NumpyToTorch, TransformReward
import torch as T
import os
import numpy as np
from collections import deque


def change_reward_type(rew):
    return rew.to(T.float)


def make_env(env_id: str, num_envs: int = 2, dtype: T.dtype = np.float32, device: T.device = T.device('cpu'), 
             env_type: str = 'async', name_prefix = None, make_video: bool = False, game_length: int = 1000, **kwargs):
        
    envs_list = []
    for i in range(num_envs):
        env = gym.make(env_id, render_mode='rgb_array' if make_video else None, **kwargs)
        env = Autoreset(env)
        env = TimeLimit(env, max_episode_steps=game_length)
        # if i == 0:
        if make_video:
            env = RecordVideo(env, video_folder='.//save_videos', fps=100, name_prefix=name_prefix)
        envs_list.append(lambda: env)
    if env_type == 'async':
        envs =  AsyncVectorEnv(envs_list) 
    else:
        envs =  SyncVectorEnv(envs_list) 
    envs = DtypeObservation(envs, dtype=dtype)
    envs = NumpyToTorch(envs, device=device)
    envs = TransformReward(envs, func=change_reward_type)
    return envs

def play_game(env_id: str, 
              agent: T.nn.Module, 
              num_games: int, 
              name_prefix: str | None = None, 
              if_rnn: bool = False,
              backstep: int | None = None,
              device: T.device = T.device('cpu'), 
              make_video: bool = False) -> T.Tensor:
    if make_video:
        if name_prefix is None:
            name_prefix = env_id
        os.makedirs('save_videos', exist_ok=True)
    env = make_env(
        env_id=env_id, num_envs=1, device=device, env_type=SyncVectorEnv, name_prefix=name_prefix, make_video=make_video, game_length=1000
        )
    
    games = 0
    total_rewards = T.zeros(1)
    state, _ = env.reset()
    list_rewards = []
    
    if if_rnn:
        state_temp_memory = deque(maxlen=backstep)
        action_dim = agent.num_actions
        prev_action = T.zeros(1, action_dim, device=device)
        for i in range(backstep):
            if i == backstep - 1:
                done = T.ones(1, 1, device=device)
            else:
                done = T.zeros(1, 1, device=device)
            state_temp_memory.append(T.cat([state, prev_action, done], dim=-1))
    
    while games < num_games:
        if if_rnn:
            stacked_state = T.stack(list(state_temp_memory), dim=-2).to(device)
            state_to_current = stacked_state
        else:
            state_to_current = state
        with T.no_grad():
            output = agent.select_action(state_to_current, temperature=0.01)
        action = output['action']
        action = action.clamp(min=-1, max=1)
        state, reward, terminated, truncated, _ = env.step(action)     
        done = T.logical_or(terminated, truncated)   
        if if_rnn:
            state = T.cat([state, action, done.unsqueeze(-1)], dim=-1)
            state_temp_memory.append(state)
        total_rewards += reward.cpu()
        if done:
            list_rewards.append(total_rewards)
            total_rewards = T.zeros(1)
            games += 1
    env.close()
    if make_video:
        print('Video saved')
    print(state[0], action)
    return np.mean(list_rewards), np.std(list_rewards)
