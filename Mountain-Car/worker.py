from helper_functions import make_env, play_game
from agent import ACAgent
from memory import Memory
import torch as T
import numpy as np
from collections import deque
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models import ICM
import os


def reshape_reward(reward, action, velocity, next_position, position):
    side_reward = - 0.1
    side_reward += 100. * action * velocity # * (10 ** (np.sign(action) != np.sign(velocity)))
    # side_reward += 100. * velocity ** 2
    side_reward += 0.5 * (next_position + 1.2) ** 2
    # side_reward += 0.1 * (abs(next_position - position) * 10) ** 2
    side_reward += 10. * np.logical_and(next_position > 0, position < 0)
    side_reward += 25. * np.logical_and(next_position > 0.2, position < 0.2)
    side_reward += 50. * np.logical_and(next_position > 0.4, position < 0.4)
    return reward + side_reward

    
class WorkerNStep:
    def __init__(self, env_id, batch_size: int = 1, timesteps: int = 5, device = 'cpu'):
        self.env_id = env_id
        self.device = device
        self.batch_size = batch_size
        self.env = make_env(env_id, batch_size=batch_size)
        n_actions = self.env.action_space.shape[-1] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
        self.agent = ACAgent(input_dim=2, output_dim=n_actions, timesteps=timesteps, hidden_dim=8, device=device)
        self.memory = Memory(maxlen=timesteps+1, device=device)
        self.timesteps = timesteps
        self.optimizer = Adam(self.agent.agent.parameters(), lr=0.0002, weight_decay=0.)
        self.optimizer.zero_grad()
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 100, T_mult=2)
        self.ICM = ICM(input_dim=2, hidden_dim=8, output_dim=1).to(device)
        self.icm_optimizer = Adam(self.ICM.parameters(), lr=0.0002, weight_decay=0.0001)
        
    def save_model(self):
        os.makedirs('outputs', exist_ok=True)
        T.save(self.agent.agent.state_dict(), 'outputs//agent_state_dict.pth')
        
    def load_model(self):
        self.agent.agent.load_state_dict(T.load('outputs//agent_state_dict.pth'))
        
    def training_step(self, entropy_reg):
        sample = self.memory.get(agg_type=T.stack)
        losses, L_I, L_F = self.agent.train(sample, entropy_reg, self.ICM)
        loss = losses[0]
        self.icm_optimizer.zero_grad()
        (L_I + L_F).backward()
        self.icm_optimizer.step()
        loss.backward()
        T.nn.utils.clip_grad_value_(self.agent.agent.parameters(), 10)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.memory.reset()
        for name, param in self.agent.agent.named_parameters():
            if T.isnan(param).any():
                raise ValueError(f"Parameter '{name}' contains NaN values.")
        return losses
    
    def action_step(self, state, temperature):
        action, log_prob, state_value, entropy = self.agent.action(state, distribution="normal", temperature=temperature, min_std=0.5)
        action = action.detach()
        next_state, reward, terminated, truncated, _ = self.env.step(action.cpu())
        position, velocity = next_state[..., 0], next_state[..., 1]
        pure_reward = reward
        reward = reshape_reward(reward, action[..., 0].cpu().numpy(), velocity, next_position=position, position=state[..., 0].detach().numpy())
        reward = T.tensor(reward, dtype=T.float32)  
        done = T.tensor(np.logical_or(terminated, truncated), dtype=T.bool)
        next_state = T.tensor(next_state, dtype=T.float32)
        items = (state, state_value, action, entropy, log_prob, reward, done)
        self.memory.push(items)
        return next_state, pure_reward, done, terminated
        
        
    def train(self):
        temperature = 1
        best_value = -100
        loss_list = [[], [], [], []]
        step = 0
        exit_count = -100
        max_steps = int(1e6)
        
        state, _ = self.env.reset()
        state = T.tensor(state, dtype=T.float32)
        
        num_games = []
        num_wins = []
        game_durations = []
        game_rewards_list = []
        total_rewards = np.zeros(self.batch_size)
        episode_step = np.zeros(self.batch_size)
        
        while step < max_steps:
            temperature = max(0.5, 2 / (1 + 0.01 * (step // 1000)))
            entropy_reg = 0.01 - (step // 1000) * (0.01 - 0.0000001) / (max_steps // 1000)
            
            state, reward, done, terminated = self.action_step(state, temperature)
            if step > 0 and step % (self.timesteps + 1) == 0:
                losses = self.training_step(entropy_reg)
                for i, loss in enumerate(losses):
                    loss_list[i].append(loss.item())
            episode_step += 1
            total_rewards = total_rewards + reward
            if done.sum() > 0:
                num_games.append(done.sum())
                num_wins.append(np.sum(terminated))
                game_durations.append(np.sum(episode_step * done.numpy()))
                game_rewards_list.append(np.sum(total_rewards * done.numpy()))
                
                total_rewards = total_rewards * (1 - done.numpy())            
                episode_step = episode_step * (1 - done.numpy())
            
            
            if step % 1000 == 0:
                self.scheduler.step()
                num_g = np.sum(num_games[-50:])
                num_g = max(num_g, 1)
                dur_g = np.sum(game_durations[-50:])
                rew_g = np.sum(game_rewards_list[-50:])
                print(f"Steps: {step}, wins {np.sum(num_wins[-50:]):.0f}, over num games {num_g:.0f}, avg duration {dur_g/num_g:.2f} "
                      f"mean rewards {rew_g/num_g:.2f} "
                    f"loss mean {np.mean(loss_list[0][-1000:]):.4f} policy loss mean {np.mean(loss_list[1][-1000:]):.4f} "
                    f"critic loss mean {np.mean(loss_list[2][-1000:]):.4f} entropy mean {np.mean(loss_list[3][-1000:]):.4f} "
                    f"best value {best_value:.2f} current learning rate {self.scheduler.get_last_lr()[0]:.6f}"
                    # f" LR {self.agent.scheduler.get_last_lr()[0]:.6f}, temperature {temperature:.4f}"
                    )
            if step % 10000 == 0:
                distribution = "normal"
                temperature = 0.01
                _, terminated = play_game(self.env_id, self.agent, distribution=distribution, temperature=temperature, single_env=True)
                if terminated: 
                    game_reward, _ = play_game(self.env_id, self.agent, distribution=distribution, temperature=temperature, make_video=False, repeat=100, single_env=True)
                    if game_reward/100 >= best_value:
                        exit_count = 0
                        best_value = game_reward/100
                        print(f"New best value {best_value:.2f}")
                        self.save_model()
                    else:
                        print(f"Average over 100 games {game_reward/100}")
                # exit_count += 1
            # if exit_count > 10:
            #     print('Not improving for 10 episodes, stopping')
            #     break
            step += 1
                
        print("Final Game")
        self.load_model()
        distribution = "normal"
        temperature = 0.01
        play_game(self.env_id, self.agent, distribution=distribution, temperature=temperature, make_video=True, repeat=100)
        