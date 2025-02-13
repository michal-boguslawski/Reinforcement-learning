import torch as T
from helper_functions import make_env
from memory import Memory
from agent import Agent
import numpy as np

T.set_printoptions(precision=6, sci_mode=False, linewidth=200)

def reshape_reward(reward, action, state, next_state, factor):
    diff = next_state - state
    side_reward = T.zeros_like(reward)
    #side_reward -= 0.05 * ((next_state[..., :1] ** 2).sum(-1) - T.pi)
    side_reward -= 0.02 * ((next_state[..., :3] ** 2).sum(-1) - 2 * T.pi)
    # side_reward += 10 * (diff[..., :3] ** 2).sum(-1)
    
    side_reward -= 0.01 * (next_state[..., 4] ** 2 - T.pi)
    side_reward += 0.01 * next_state[..., 3] * T.abs(next_state[..., 3]) 
    # side_reward += 0.01 * (next_state[..., 6:] ** 2).sum(-1)
    
    # side_reward -= 0.01 * action[..., 0] * action[..., 1]
    # side_reward -= 0.05 * action[..., 0] * next_state[..., 1] / T.pi
    # side_reward -= 0.1 * action[..., 1] * next_state[..., 2] / T.pi
    return reward + side_reward * (1 - factor)


class Worker:
    def __init__(self, env_id, input_dim, hidden_dim, timesteps: int = 32, learning_rate: float = 0.0002, max_steps=int(1e8), device: T.device = T.device('cpu')):
        self.env = make_env(env_id, device=device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.device = device
        output_dim = self.env.action_space.shape[-1] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
        self.output_dim = output_dim
        self.agent = Agent(input_dim, output_dim, hidden_dim, num_layers=3, learning_rate=learning_rate, timesteps=timesteps, device=device)
        self.memory = Memory(maxlen=timesteps+1)
        self.load_model()
        
    def save_model(self):
        self.agent.save_model()
        
    def load_model(self):
        self.agent.load_model()
        
    def action_step(self, state, hx, step):
        factor = step/self.max_steps
        action, log_prob, hx_out = self.agent.action(state, hx)
        used_action = T.tanh(action)
        next_state, reward, terminated, truncated, _ = self.env.step(used_action)
        reward = T.tensor([reward,], dtype=T.float32, device=self.device)
        terminated = T.tensor([terminated,], dtype=T.bool, device=self.device)
        truncated = T.tensor([truncated,], dtype=T.bool, device=self.device)
        done = T.logical_or(terminated, truncated)
        reward_to_train = reshape_reward(reward, used_action, state, next_state, factor)
        items = (action, state, log_prob, hx_out, reward_to_train, done)
        self.memory.push(items)
        return next_state, hx_out, done, reward, used_action
    
    def training_step(self):
        batch = self.memory.get(agg_type=T.stack)
        losses = self.agent.train(batch)
        self.memory.reset()
        return losses
        
    def train(self):
        step = 0
        losses_list = []
        total_reward = T.zeros(1)
        list_rewards = []
        
        state, _ = self.env.reset()
        hx = T.zeros(self.hidden_dim).to(self.device)
        done = False
        
        while step < self.max_steps:
            state, hx, done, reward, action = self.action_step(state, hx, step)
            
            if step > 0 and step % (self.timesteps + 1) == 0:
                losses = self.training_step()
                losses_list.extend(losses)
            total_reward = total_reward + reward.cpu()
            if done:
                list_rewards.append(total_reward)
                print(f"{total_reward}, {state.cpu()}, {action.cpu()}")
                total_reward = T.zeros(1)
                hx = T.zeros(self.hidden_dim).to(self.device)
            if step % 1000 == 0:
                print(f"Step {step}, loss {np.mean(losses_list[-1000:]):.6f}, rewards {np.mean(list_rewards[-100:]):.4f}")
                
            if step % 10000 == 0:
                self.save_model()
            step += 1
                