from helper_functions import make_env, play_game
from agent import ACAgent
from memory import Memory
import torch as T
import numpy as np
from collections import deque
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


def reshape_reward(reward, velocity, next_position, position):
    side_reward = - 0.1 \
        + np.sign(abs(velocity) - 0.03) * (100 * abs(velocity) - 3) ** 2 / 100 \
        + 0.1 * (next_position + 1.2) ** 2 \
        + 1 * (abs(next_position - position) * 10) ** 2 \
        + 10. * int(next_position > 0 and position < 0) \
        + 25. * int(next_position > 0.2 and position < 0.2) \
        + 50. * int(next_position > 0.4 and position < 0.4)
    return reward + side_reward

    
class WorkerNStep:
    def __init__(self, env_id, timesteps: int = 5, device = 'cpu', episodes: int = 1000):
        self.env_id = env_id
        self.device = device
        self.env = make_env(env_id)
        n_actions = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
        self.agent = ACAgent(input_dim=2, output_dim=n_actions, timesteps=timesteps, hidden_dim=8, device=device)
        self.memory = Memory(maxlen=timesteps+1)
        self.timesteps = timesteps
        self.episodes = episodes
        self.optimizer = Adam(self.agent.agent.parameters(), lr=0.0002, weight_decay=0.)
        self.scheduler = CosineAnnealingLR(self.optimizer, episodes)
        self.optimizer.zero_grad()
        
    def save_model(self):
        os.makedirs('outputs', exist_ok=True)
        T.save(self.agent.agent.state_dict(), 'outputs//agent_state_dict.pth')
        
    def training_step(self, entropy_reg):
        sample = self.memory.get(agg_type=T.cat)
        losses = self.agent.train(sample, entropy_reg)
        loss = losses[0]
        loss.backward()
        T.nn.utils.clip_grad_value_(self.agent.agent.parameters(), 1)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.memory.reset()
        for name, param in self.agent.agent.named_parameters():
            if T.isnan(param).any():
                raise ValueError(f"Parameter '{name}' contains NaN values.")
        return losses
    
    def action_step(self, state, temperature):
        action, log_prob, state_value, entropy = self.agent.action(state, distribution="normal", temperature=temperature, min_std=0.1)
        next_state, reward, terminated, truncated, _ = self.env.step([float(action),])
        position, velocity = next_state
        pure_reward = reward
        reward = reshape_reward(reward, velocity, state[0, 0], position)
        reward = T.tensor([reward], dtype=T.float32)  
        done = T.tensor([terminated or truncated, ], dtype=T.bool)
        next_state = T.tensor(next_state, dtype=T.float32)
        next_state = next_state.unsqueeze(0)
        items = (state_value, entropy, log_prob, reward, done)
        self.memory.push(items)
        return next_state, pure_reward, done, terminated
        
        
    def train(self):
        temperature = 1
        best_value = -100
        loss_list = [[], [], [], []]
        list_rewards = []
        step = 0
        list_wins = []
        exit_count = -100
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            state = T.tensor(state, dtype=T.float32)
            state = state.unsqueeze(0)
            done = False
            episode_step = 0
            total_reward = 0
            temperature = max(0.5, 2 / (1 + 0.01 * episode))
            entropy_reg = 0.01 - episode * (0.01 - 0.0000001) / self.episodes
            while not done:
                state, reward, done, terminated = self.action_step(state, temperature)
                total_reward += reward
                if step > 0 and step % (self.timesteps + 1) == 0:
                    losses = self.training_step(entropy_reg)
                    for i, loss in enumerate(losses):
                        loss_list[i].append(loss.item())
                step += 1
                episode_step += 1
            # print(f"Duration: {episode_step:<5}, result: {terminated:<2}, last action: {float(action):<6.2f}, final episode reward {total_reward:<6.2f}, final position {position:.4f}")
            list_wins.append(int(terminated))
            list_rewards.append(total_reward)
            self.scheduler.step()
            # self.agent.step_scheduler()
                # print(self.agent.local_agent.init_norm)
            if episode % 10 == 0:
                print(f"Episode no {episode}, steps: {step}, total rewards {np.mean(list_rewards[-50:]):.1f} "
                    f"loss mean {np.mean(loss_list[0][-1000:]):.4f} policy loss mean {np.mean(loss_list[1][-1000:]):.4f} "
                    f"critic loss mean {np.mean(loss_list[2][-1000:]):.4f} entropy mean {np.mean(loss_list[3][-1000:]):.4f} "
                    f"wins last 10 {np.sum(list_wins[-50:]):.0f}, best value {best_value:.2f}"
                    # f" LR {self.agent.scheduler.get_last_lr()[0]:.6f}, temperature {temperature:.4f}"
                    )
            if episode % 100 == 0:
                distribution = "normal"
                temperature = 0.01
                _, terminated = play_game(self.env_id, self.agent, distribution=distribution, temperature=temperature)
                if terminated: 
                    game_rewards = 0
                    for _ in range(100):
                        game_reward, _ = play_game(self.env_id, self.agent, distribution=distribution, temperature=temperature, make_video=False)
                        game_rewards += game_reward
                    if game_rewards/100 >= best_value:
                        exit_count = 0
                        best_value = game_rewards/100
                        print(f"New best value {best_value:.2f}")
                        self.save_model()
                exit_count += 1
            if exit_count > 10:
                print('Not improving for 10 episodes, stopping')
                break
                
        print("Final Game")
        play_game(self.env_id, self.agent)
        