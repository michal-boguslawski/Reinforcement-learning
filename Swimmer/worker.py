import torch as T
from helper_functions import make_env, play_game
from memory import Memory
from agent import Agent
import numpy as np

T.set_printoptions(precision=6, sci_mode=False, linewidth=200)

def reshape_reward(reward, action, state, next_state, factor):
    diff = next_state - state
    side_reward = T.zeros_like(reward)
    side_reward -= 0.05 * ((next_state[..., :1] ** 2).sum(-1) - T.pi)
    side_reward -= 0.04 * ((next_state[..., :2] ** 2).sum(-1) - T.pi)
    side_reward -= 0.01 * ((next_state[..., :3] ** 2).sum(-1) - 2 * T.pi)
    # side_reward += 10 * (diff[..., :3] ** 2).sum(-1)
    
    side_reward -= 0.01 * (next_state[..., 4] ** 2 - T.pi)
    side_reward += 0.01 * next_state[..., 3] * T.abs(next_state[..., 3]) 
    # side_reward += 0.01 * (next_state[..., 6:] ** 2).sum(-1)
    
    # side_reward -= 0.01 * action[..., 0] * action[..., 1]
    side_reward -= 0.02 * action[..., 0] * state[..., 1] * T.abs(action[..., 0] * state[..., 1])
    side_reward -= 0.05 * action[..., 1] * state[..., 2] * T.abs(action[..., 1] * state[..., 2])
    return reward + side_reward * (1 - factor)


class Worker:
    def __init__(self, env_id, hidden_dim, num_envs: int = 2, timesteps: int = 32, learning_rate: float = 0.0002, max_steps=int(1e8), device: T.device = T.device('cpu')):
        self.env_id = env_id
        self.env = make_env(env_id, num_envs=num_envs, device=device)
        self.num_envs = num_envs
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.device = device
        input_dim = self.env.observation_space.shape[1]
        self.input_dim = input_dim
        output_dim = self.env.action_space.shape[-1] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
        self.output_dim = output_dim
        self.agent = Agent(input_dim, output_dim, hidden_dim, num_layers=3, learning_rate=learning_rate, timesteps=timesteps, device=device)
        self.memory = Memory(maxlen=timesteps+1)
        # self.load_model()
        
    def save_model(self):
        self.agent.save_model()
        
    def load_model(self):
        self.agent.load_model()
        
    def action_step(self, state, hx, step):
        factor = step/self.max_steps
        action, log_prob, hx_out = self.agent.action(state, hx)
        used_action = T.tanh(action)
        next_state, reward, terminated, truncated, _ = self.env.step(used_action)
        # reward = T.tensor([reward,], dtype=T.float32, device=self.device)
        # terminated = T.tensor([terminated,], dtype=T.bool, device=self.device)
        # truncated = T.tensor([truncated,], dtype=T.bool, device=self.device)
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
        num_games = []
        list_rewards = []
        total_reward = T.zeros(self.num_envs)
        
        state, _ = self.env.reset()
        hx = T.zeros(self.num_envs, self.hidden_dim).to(self.device)
        done = False
        
        while step < self.max_steps:
            state, hx, done, reward, action = self.action_step(state, hx, step)
            
            if step > 0 and step % (self.timesteps + 1) == 0:
                losses = self.training_step()
                losses_list.extend(losses)
            total_reward = total_reward + reward.cpu()
            if done.any():
                list_rewards.append((total_reward * done.cpu()).sum())
                num_games.append(done.cpu().sum())
                print(f"{total_reward.mean():.6f}, {state.cpu().mean(0)}, {action.cpu().mean(0)}")
                total_reward = T.zeros(self.num_envs)
                hx = T.zeros(self.num_envs, self.hidden_dim).to(self.device)
            if step % 1000 == 0:
                mean_rewards = np.sum(list_rewards[-100:]) / np.sum(num_games[-100:])
                print(f"Step {step}, loss {np.mean(losses_list[-1000:]):.6f}, rewards {mean_rewards:.4f} "
                      f"num games {np.sum(num_games[-100:]):.0f}")
                
            if step % 10000 == 0:
                step_result = play_game(env_id=self.env_id, hx_start=T.zeros(1, self.hidden_dim), agent=self.agent, num_games=1, name_prefix=f"Game-{step//10000}", device=self.device, make_video=True)
                print(f"Game {step//10000}, result {step_result}")
                self.save_model()
            step += 1
                