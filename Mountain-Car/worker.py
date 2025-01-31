from helper_functions import make_env, play_game
from agent import ACAgent
from memory import Memory
import torch as T
import numpy as np
from collections import deque

    
class WorkerNStep:
    def __init__(self, env_id, batch_size: int = 128, timesteps: int = 5, device = 'cpu'):
        self.env_id = env_id
        self.device = device
        self.env = make_env(env_id)
        n_actions = self.env.action_space.n
        self.agent = ACAgent(n_observations=4, n_actions=n_actions, timesteps=timesteps, hidden_dim=32, device=device)
        self.memory = Memory(maxlen=100000)
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
        for episode in range(episodes):
            done = False
            total_reward = 0
            while not done:
                temperature = 10 / np.log(episode + 1e-8)
                action = self.agent.action(state, strategy="softmax", temperature=temperature)
                next_state, reward, terminated, truncated, _ = self.env.step(float(action))
                action = T.tensor([float(action), ], dtype=T.float32)
                reward = T.tensor([reward, ], dtype=T.float32)
                terminated = T.tensor([terminated, ], dtype=T.bool)
                total_reward += reward
                next_state = T.tensor(next_state, dtype=T.float32)
                next_state = next_state.unsqueeze(0)
                items = (
                    next_state, 
                    action, 
                    reward, 
                    terminated
                    )
                self.append_to_buffers(items)
                # td_error = self.agent.calc_one_step_td_error(state, next_state, action, reward, terminated)
                self.priority_buffer.append(1)
                if step >= self.timesteps:
                    self.memory.push(
                        self.prepare_to_buffer(), 
                        np.sum(list(self.priority_buffer))
                        )
                if step % 4 == 0 and step >= self.batch_size + self.timesteps:
                    sample = self.memory.sample(sample_size=self.batch_size, agg_type=T.stack)
                    loss = self.agent.train(sample)
                    loss_list.append(loss)
                step += 1
                done = terminated or truncated
                state = next_state
            
            list_rewards.append(total_reward)
            state, _ = self.env.reset()
            state = T.tensor(state, dtype=T.float32)
            state = state.unsqueeze(0)
            # self.agent.step_scheduler()
            if episode % 10 == 0:
                print(f"Episode no {episode}, steps: {step}, total rewards {np.mean(list_rewards[-10:]):.1f} loss mean {np.mean(loss_list[-10:]):.4f}"
                      f" LR {self.agent.scheduler.get_last_lr()[0]:.6f}, temperature {temperature:.4f}"
                      )
                # print(self.agent.local_agent.init_norm)
            if episode % 100 == 0:
                strategy = "softmax"
                temperature = 0.01
                play_game(self.env_id, self.agent, strategy=strategy, temperature=temperature)
        print("Final Game")
        play_game(self.env_id, self.agent)
        T.save(self.agent.local_agent.state_dict(), 'local_agent_state_dict.pth')
        T.save(self.agent.target_agent.state_dict(), 'target_agent_state_dict.pth')
        