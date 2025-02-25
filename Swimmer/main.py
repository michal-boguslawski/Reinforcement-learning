from helper_functions import make_env, play_game
from models import PPO
from memory import Memory
import torch as T
import torch.nn as nn
import numpy as np
import os

T.set_printoptions(precision=4, sci_mode=False, linewidth=200)

def reshape_reward(reward, state, next_state, action, factor):
    side_reward = T.zeros_like(reward)
    
    side_reward -= 0.01 * T.square(next_state[..., :1]).sum(-1)
    side_reward -= 0.1 * T.square(next_state[..., :2]).sum(-1) * factor 
    side_reward -= 0.1 * T.square(next_state[..., :3]).sum(-1) * factor
    side_reward -= 0.1 * (next_state[..., 1] * next_state[..., 2]).clamp(min=0) * factor
    
    return reward + side_reward

timesteps = 4096
ppo_epochs = 10
max_timesteps = int(1e6)
num_envs = 16
gru_dims = 32
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
os.environ["MUJOCO_GL"] = "egl" if T.cuda.is_available() else 'osmesa'
env_id = 'Swimmer-v5'
envs = make_env('Swimmer-v5', num_envs=num_envs, device=device)
action_dim = envs.action_space.shape[-1]
input_dim = envs.observation_space.shape[-1]
ac_model = PPO(num_actions=action_dim, 
               input_dim=input_dim, 
               hidden_dims=32, 
               gru_dims=gru_dims, 
               if_gru=False, 
               final_activation=nn.Tanh(), 
               lr=0.0003,
               device=device)
memory = Memory(timesteps)

state, _ = envs.reset()

step = 0
total_rewards = 0
factor = 1
list_rewards = []
list_num_games = []
list_wins = []
hx = T.zeros(num_envs, gru_dims)
while step <= max_timesteps:
    with T.no_grad():
        output = ac_model.select_action(state, hx=hx)
    action = output['action']
    hx = output['hx']
    next_state, reward, terminated, truncated, _ = envs.step(action)
    reward_to_model = reshape_reward(reward, state, next_state, action, factor)
    done = T.logical_or(terminated, truncated)
    item = (action, state, hx, output['logprob'], output['value'], reward_to_model, terminated, truncated)
    memory.push(item)
    
    total_rewards += reward
    if done.any():
        hx[done] = 0
        print(state.mean(0)[:3].cpu(), total_rewards[done].cpu().mean())
        list_rewards.append(total_rewards[done].cpu().sum())
        list_num_games.append(done.cpu().sum())
        list_wins.append(terminated.cpu().sum())
        if (total_rewards > 100).any():
            factor *= 0.999 ** (total_rewards[total_rewards > 100].sum()/num_envs)
        total_rewards = total_rewards * T.logical_not(done)
    
    state = next_state
    
    if step % timesteps == 0 and step > 0:
        sum_rewards = np.sum(list_rewards[-100:])
        num_games = np.sum(list_num_games[-100:])
        wins = np.sum(list_wins[-100:])
        print(f"Steps {step}, mean rewards {sum_rewards/num_games:.2f}, mean wins {wins/num_games:.4f}, over games {num_games} "
              f"factor {factor:.4f}, std ")
        print({output['cov_matrix'].mean(0).cpu()})
        batch = memory.get(agg_type=T.stack)
        
        for _ in range(ppo_epochs):
            ac_model.update(batch)

        ac_model.save_model()
        
    if step % 50000 == 0:
        temp_result = play_game(env_id, ac_model, num_games=1, device=device, name_prefix=env_id + f"_{step//50000}", make_video=True)
        print(temp_result)
    
    step += 1
    
envs.close()
ac_model.save_model()
_ = play_game(env_id, ac_model, num_games=1, device=device, name_prefix=env_id, make_video=True)
rewards = play_game(env_id, ac_model, num_games=50, device=device, name_prefix=env_id, make_video=False)
print(rewards)
