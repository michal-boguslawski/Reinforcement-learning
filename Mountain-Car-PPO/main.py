from helper_functions import make_env, play_game
from models import PPO
from memory import Memory
import torch as T
import torch.nn as nn
import numpy as np

T.set_printoptions(precision=4, sci_mode=False, linewidth=200)

def reshape_reward(reward, state, next_state, action, factor):
    side_reward = T.zeros_like(reward)
    
    side_reward += 1000 * T.square(next_state[..., 1]) + 10 * T.abs(next_state[..., 1]) * factor
    side_reward += 5 * (state[..., 0] < 0) * (next_state[..., 0] > 0) * factor
    side_reward += 10 * (state[..., 0] < 0.2) * (next_state[..., 0] > 0.2) * factor
    side_reward += 10 * (state[..., 1] * action[..., 0]).clamp(max=0.01) * factor
    side_reward += 0.1 * T.abs(action[..., 0]) * factor
    side_reward -= (T.square(action).sum(-1) + T.abs(action).sum(-1)) * (1 - factor)
    
    return reward + side_reward

timesteps = 1024
ppo_epochs = 3
max_timesteps = int(1e6)
num_envs = 16
gru_dims = 32
envs = make_env('MountainCarContinuous-v0', num_envs=num_envs)
action_dim = envs.action_space.shape[-1]
input_dim = envs.observation_space.shape[-1]
ac_model = PPO(num_actions=action_dim, input_dim=input_dim, hidden_dims=16, gru_dims=gru_dims, if_gru=False, final_activation=nn.Tanh(), lr=0.0003)
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
    state = next_state
    
    total_rewards += reward
    if done.any():
        hx[done] = 0
        list_rewards.append(total_rewards[done].sum())
        list_num_games.append(done.sum())
        list_wins.append(terminated.sum())
        if terminated.any():
            factor *= 0.99 ** (terminated.sum()/num_envs)
        total_rewards = total_rewards * T.logical_not(done)
    
    if step % timesteps == 0 and step > 0:
        sum_rewards = np.sum(list_rewards[-100:])
        num_games = np.sum(list_num_games[-100:])
        wins = np.sum(list_wins[-100:])
        print(f"Steps {step}, mean rewards {sum_rewards/num_games:.2f}, mean wins {wins/num_games:.4f}, over games {num_games}, std {output['cov_matrix'].mean():.4f} "
              f"factor {factor:.4f}")
        batch = memory.get(agg_type=T.stack)
        
        for _ in range(ppo_epochs):
            ac_model.update(batch, factor=2 * factor - 1)

        ac_model.save_model()
    
    step += 1
    
envs.close()
ac_model.save_model()
_ = play_game('MountainCarContinuous-v0', ac_model, num_games=1, name_prefix='PPO_mountain_car', make_video=True)
rewards = play_game('MountainCarContinuous-v0', ac_model, num_games=50, name_prefix='PPO_mountain_car', make_video=False)
print(rewards)
