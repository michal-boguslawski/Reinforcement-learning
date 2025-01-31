from helper_functions import make_env, play_game
from agent import QAgent, EligibilityQAgent
from memory import Memory
import torch as T
import numpy as np
from collections import deque
T.manual_seed(42)


class Worker:
    def __init__(self, env_id):
        self.env_id = env_id
        self.env = make_env(env_id)
        n_actions = self.env.action_space.n
        self.agent = QAgent(n_observations=4, n_actions=n_actions)
        self.memory = Memory(maxlen=10000)
        self.batch_size = 128
        
    def train(self, episodes: int = 1000):
        loss_list = []
        list_rewards = []
        list_steps = []
        step = 0
        for episode in range(episodes):
            done = False
            total_reward = 0
            state, _ = self.env.reset()
            state = T.tensor(state, dtype=T.float32)
            state = state.unsqueeze(0)
            while not done:
                action = self.agent.action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(int(action))
                total_reward += reward
                next_state = T.tensor(next_state, dtype=T.float32)
                next_state = next_state.unsqueeze(0)
                self.memory.push(
                    (
                        state,
                        next_state,
                        action,
                        T.tensor([reward, ], dtype=T.float32),
                        T.tensor([terminated, ], dtype=T.bool)
                    )
                )
                if step % 4 == 0 and step >= self.batch_size:
                    sample = self.memory.sample(self.batch_size)
                    loss = self.agent.train(sample)
                    loss_list.append(loss)
                step += 1
                done = terminated or truncated
                state = next_state
            
            list_rewards.append(total_reward)
            if episode % 100 == 0:
                print(f"Episode no {episode}, steps: {step}, total rewards {np.mean(list_rewards[-100:]):.4f} loss mean {np.mean(loss_list[-100:]):.4f}")
        self.play_game()
    
class WorkerNStep:
    def __init__(self, env_id, batch_size: int = 128, timesteps: int = 5, device = 'cpu'):
        self.env_id = env_id
        self.device = device
        self.env = make_env(env_id)
        n_actions = self.env.action_space.n
        self.agent = EligibilityQAgent(n_observations=4, n_actions=n_actions, timesteps=timesteps, hidden_dim=16, device=device)
        self.memory = Memory(maxlen=100000, weight=0.9999)
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.buffers = [deque(maxlen=timesteps+int(i==0)) for i in range(4)]
        self.priority_buffer = deque(maxlen=timesteps)
        
    def append_to_buffers(self, items):
        for i, x in enumerate(items):
                self.buffers[i].append(x)
        
    def reset_buffer(self, state):
        for x in self.buffers:
            x.clear()        
        self.buffers[0].append(state)
        
    def prepare_to_buffer(self):
        return [T.cat(list(x)) for x in self.buffers]
        
        
    def train(self, episodes: int = 1000):
        temperature = 1
        loss_list = []
        list_rewards = []
        step = 0
        state, _ = self.env.reset()
        state = T.tensor(state, dtype=T.float32)
        state = state.unsqueeze(0)
        self.reset_buffer(state)
        batch_size_factor = 1
        for episode in range(episodes):
            done = False
            total_reward = 0
            episode_step = 0
            while not done:
                temperature = 2
                action = self.agent.action(state, strategy="softmax", temperature=temperature)
                next_state, reward, terminated, truncated, _ = self.env.step(int(action))
                total_reward += reward
                action = T.tensor([int(action), ], dtype=T.int64)
                reward = T.tensor(
                    [reward 
                     - 0.2 * abs(4 * next_state[0] / 4.8) ** 2 
                     - 0.2 * abs(4 * next_state[2] / 0.418) ** 2
                     - 0.4 * (16 * next_state[0] / 4.8 * next_state[2] / 0.418) ** 2, ], dtype=T.float32
                    )
                terminated = T.tensor([terminated, ], dtype=T.bool)
                next_state = T.tensor(next_state, dtype=T.float32)
                next_state = next_state.unsqueeze(0)
                items = (
                    next_state,
                    action, 
                    reward, 
                    terminated
                    )
                self.append_to_buffers(items)
                td_error = self.agent.calc_one_step_td_error(state, next_state, action, reward, terminated)
                self.priority_buffer.append(td_error)
                if step >= self.timesteps:
                    self.memory.push(self.prepare_to_buffer(), np.log(1+np.sum(list(self.priority_buffer))))
                if step % 4 == 0 and step >= self.batch_size + self.timesteps:
                    sample = self.memory.sample(sample_size=int(self.batch_size*batch_size_factor), agg_type=T.stack)
                    loss = self.agent.train(sample)
                    loss_list.append(loss)
                step += 1
                episode_step += 1
                done = terminated or truncated
                state = next_state
            
            list_rewards.append(total_reward)
            state, _ = self.env.reset()
            state = T.tensor(state, dtype=T.float32)
            state = state.unsqueeze(0)
            self.agent.step_scheduler()
            if episode % 10 == 0:
                print(f"Episode no {episode}, steps: {step}, avg duration {np.mean(list_rewards[-10:]):.1f} loss mean {np.mean(loss_list[-10:]):.4f}"
                      f" LR {self.agent.scheduler.get_last_lr()[0]:.6f}, temperature {temperature:.4f}"
                      )
            if episode % 100 == 0:
                strategy = "softmax"
                game_temperature = 0.01
                _ = play_game(self.env_id, self.agent, strategy=strategy, temperature=game_temperature)
        print("Final Game")
        play_game(self.env_id, self.agent)
        T.save(self.agent.local_agent.state_dict(), 'local_agent_state_dict.pth')
        T.save(self.agent.target_agent.state_dict(), 'target_agent_state_dict.pth')
        