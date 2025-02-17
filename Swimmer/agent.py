import torch as T
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import HuberLoss
from models import SwimmerModel
from torch.distributions.multivariate_normal import MultivariateNormal
import os


def get_discounted_table(n, v):
    tab = T.arange(n).unsqueeze(1) - T.arange(n)
    tab = v ** tab
    return tab.tril()


class Agent:
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers: int = 3, learning_rate: float = 0.0002, timesteps: int = 5, device: T.device = 'cpu', 
                 lambda_: float = 0.9, gamma_: float = 0.95, epsilon_: float = 0.2, beta_: float = 0.1, distribution: str = "normal"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.device = device
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.epsilon_ = epsilon_
        self.beta_ = beta_
        self.distribution = distribution
        self.model = SwimmerModel(input_dim, output_dim, hidden_dim, num_layers).to(device)
        self.model_old = SwimmerModel(input_dim, output_dim, hidden_dim, num_layers).to(device)
        self.update_model_old()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        self.loss_fn = HuberLoss(reduction='none')
        self.discounted_table = get_discounted_table(timesteps, self.gamma_ * self.lambda_).to(device)
        self.upper_triangle = T.ones(timesteps, timesteps).triu().to(device)
        
    def save_model(self):
        os.makedirs('outputs', exist_ok=True)
        T.save(self.model.state_dict(), 'outputs//agent_state_dict.pth')
    
    def load_model(self):
        self.model.load_state_dict(T.load('outputs//agent_state_dict.pth', weights_only=True))
        self.update_model_old()
        
    def update_model_old(self):
        self.model_old.load_state_dict(self.model.state_dict())
        
    def action(self, state, hx):
        with T.no_grad():
            action_mean, action_std, _, hx_out = self.model_old(state, hx)
        if self.distribution == "normal":
            dist = MultivariateNormal(action_mean, action_std + 1e-6)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, hx_out
    
    def get_errors(self, rewards, state_values, dones, following_dones):
        next_state_values = state_values[:, 1:].clone().detach()
        state_values = state_values[:, :-1]
        td_errors = rewards + self.gamma_ * next_state_values * (1 - dones) - state_values
        td_errors = td_errors * (1 - following_dones + dones)
        
        discounted_td_errors = T.matmul(td_errors, self.discounted_table)
        return discounted_td_errors
        
        
    def train_main(self, states, hxs, actions, old_log_probs, rewards, dones):
        action_mean, action_std, state_values, _ = self.model(states, hxs)
        action_mean = action_mean[:, :-1]
        action_std = action_std[:, :-1]
        state_values = state_values.squeeze(-1)
        dist = MultivariateNormal(action_mean, action_std.clamp(min=0) + 1e-6)
        log_probs = dist.log_prob(actions)
        entropys = dist.entropy()
        
        following_dones = T.matmul(dones, self.upper_triangle)
        discounted_td_errors = self.get_errors(rewards, state_values, dones, following_dones)
        
        r_t = T.exp(log_probs - old_log_probs)
        surr1 = discounted_td_errors.detach() * r_t
        surr2 = discounted_td_errors.detach() * r_t.clamp(min=1-self.epsilon_, max=1+self.epsilon_)
        
        policy_loss = - T.minimum(surr1, surr2) # + 0.5 * F.kl_div(log_probs, old_log_probs, reduction='none', log_target=True)
        
        critic_loss = self.loss_fn(discounted_td_errors, T.zeros_like(discounted_td_errors))
        
        loss = policy_loss + 0.5 * critic_loss - 0.001 * entropys
        loss = loss.clamp(min=-1000, max=1000)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        # T.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()
        
        
        return loss.item(), log_probs.detach()
        
        
    
    def train(self, batch, ppo_epochs: int = 10):
        losses = []
        
        actions, states, old_log_probs, hxs, rewards, dones = batch
        dones = dones[:, :-1].to(T.float).squeeze(-1)
        actions = actions[:, :-1]
        old_log_probs = old_log_probs[:, :-1]
        # hxs = hxs[:-1]
        rewards = rewards[:, :-1].squeeze(-1)
        mean_rewards = rewards.mean()
        std_rewards = rewards.std()
        standardized_rewards = (rewards - mean_rewards)/std_rewards
        
        
        for _ in range(ppo_epochs):
            loss, old_log_probs_ = self.train_main(states, hxs, actions, old_log_probs, standardized_rewards, dones)
            losses.append(loss)
        self.update_model_old()
        return losses
        