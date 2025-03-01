from helper_functions import make_env, play_game
from models import PPO
from memory import Memory
import torch as T
import torch.nn as nn
import numpy as np
import os
from collections import deque

T.set_printoptions(precision=4, sci_mode=False, linewidth=200)

def reshape_reward(reward, state, next_state, action, factor):
    side_reward = T.zeros_like(reward)
    
    side_reward -= 0.02 * T.square(next_state[..., :1]).sum(-1)
    side_reward -= 0.1 * T.square(next_state[..., :2]).sum(-1) * factor 
    side_reward -= 0.1 * T.square(next_state[..., :3]).sum(-1) * factor
    side_reward -= 0.1 * (next_state[..., 1] * next_state[..., 2]).clamp(min=0) * factor
    
    return reward + side_reward


device = T.device('cuda' if T.cuda.is_available() else 'cpu')
os.environ["MUJOCO_GL"] = "egl" if T.cuda.is_available() else 'osmesa'
env_id = 'Swimmer-v5'
num_envs = 16
envs = make_env(env_id, num_envs=num_envs, device=device)

memory_timesteps = 1024
train_timesteps = 256
ppo_epochs = 10
max_timesteps = int(1e6)
gru_dims = 32
hidden_dims = 32
backstep = 5
if_rnn = True
factor_treshold = 200
print_log_step = 10000
save_video_step = 50000

action_dim = envs.action_space.shape[-1]
input_dim = envs.observation_space.shape[-1]
ac_model = PPO(num_actions=action_dim,
               input_dim=input_dim+action_dim+1 if if_rnn else input_dim,
               hidden_dims=hidden_dims, 
               gru_dims=gru_dims, 
               if_rnn=if_rnn, 
               final_activation=nn.Tanh(), 
               lr=0.001,
               backstep=backstep,
               device=device)
memory = Memory(memory_timesteps)
if if_rnn:
    state_temp_memory = deque(maxlen=backstep)

state, _ = envs.reset()
if if_rnn:
    prev_action = T.zeros(num_envs, action_dim, device=device)
    for i in range(backstep):
        if i == backstep - 1:
            done = T.ones(num_envs, 1, device=device)
        else:
            done = T.zeros(num_envs, 1, device=device)
        state_temp_memory.append(T.cat([state, prev_action, done], dim=-1))

step = 0
total_rewards = 0
factor = 1
list_rewards = []
list_num_games = []
list_wins = []
while step <= max_timesteps:
    if if_rnn:
        stacked_state = T.stack(list(state_temp_memory), dim=-2).to(device)
        state_to_current = stacked_state
    else:
        state_to_current = state
    with T.no_grad():
        output = ac_model.select_action(state_to_current)
    action = output['action']
    next_state, reward, terminated, truncated, _ = envs.step(action)
    reward_to_model = reshape_reward(reward, state, next_state, action, factor ** 2)
    done = T.logical_or(terminated, truncated)
    state = next_state
    if if_rnn:
        state_to_memory = T.cat([state, action, done.unsqueeze(-1)], dim=-1)
        state_temp_memory.append(state_to_memory)
    
    item = (action, state_to_current, output['logprob'], output['value'], reward_to_model, terminated, truncated)
    memory.push(item)
    
    total_rewards += reward
    if done.any():
        print(state.mean(0)[:3].cpu(), total_rewards[done].cpu().mean())
        list_rewards.append(total_rewards[done].cpu().sum())
        list_num_games.append(done.cpu().sum())
        list_wins.append(terminated.cpu().sum())
        if (total_rewards > factor_treshold).any():
            factor *= 0.99 ** ((total_rewards[total_rewards > factor_treshold]/factor_treshold).sum()/num_envs)
        total_rewards = total_rewards * T.logical_not(done)
    
    
    if step % train_timesteps == 0 and step > 0:
        batch = memory.get(agg_type=T.stack, length=train_timesteps)
        
        ac_model.update(batch, ppo_epochs=ppo_epochs, factor=factor)

        ac_model.save_model()
        
    if step % print_log_step == 0:
        sum_rewards = np.sum(list_rewards[-100:])
        num_games = np.sum(list_num_games[-100:])
        wins = np.sum(list_wins[-100:])
        print(f"Steps {step}, mean rewards {sum_rewards/num_games:.2f}, mean wins {wins/num_games:.4f}, over games {num_games} "
              f"lr {ac_model.scheduler.get_last_lr()[0]:.6f}, factor {factor:.4f}, std ")
        print({output['cov_matrix'].mean(0).cpu()})
    
    if step % save_video_step == 0:
        temp_result = play_game(env_id, 
                                ac_model, 
                                num_games=1, 
                                if_rnn=if_rnn, 
                                backstep=backstep,
                                device=device, 
                                name_prefix=env_id + f"_{step//save_video_step}", 
                                make_video=True)
        print(temp_result)
        
    if factor < 0.5:
        train_timesteps = memory_timesteps
        ppo_epochs = 20
    
    step += 1
    
envs.close()
ac_model.save_model()
_ = play_game(env_id,
            ac_model, 
            num_games=1, 
            if_rnn=if_rnn, 
            backstep=backstep,
            device=device, 
            name_prefix=env_id, 
            make_video=True)
rewards = play_game(env_id, 
                    ac_model, 
                    num_games=50, 
                    if_rnn=if_rnn, 
                    backstep=backstep,
                    device=device, 
                    name_prefix=env_id, 
                    make_video=False)
print(rewards)
