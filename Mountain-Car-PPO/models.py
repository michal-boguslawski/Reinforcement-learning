import torch as T
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.nn import HuberLoss
import os

class ActorCritic(nn.Module):
    def __init__(self, num_actions, hidden_dims: int = 16, num_layers: int = 2, epsilon_: float = 0.2, final_activation: nn.Module | None = None, entropy_reg: float = -0.01):
        super(ActorCritic, self).__init__()
        
        layers = []
        for _ in range(1):
            layers.append(nn.LazyLinear(hidden_dims * 2))
            layers.append(nn.Tanh())
        self.mlp_extractor = nn.Sequential(*layers)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LazyLinear(hidden_dims))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dims, num_actions))
        if final_activation:
            layers.append(final_activation)
        self.actor = nn.Sequential(*layers)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LazyLinear(hidden_dims))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dims, 1))
        self.critic = nn.Sequential(*layers)
        
        self.log_std = nn.Parameter(T.zeros(num_actions) - 0.7)
        self.final_activation = final_activation
        
        self.epsilon_ = epsilon_
        self.entropy_reg = entropy_reg
        self.critic_loss_fn = HuberLoss()
        
    def forward(self, input, temperature: float = 1.):
        x = self.mlp_extractor(input)
        actor = self.actor(x)
        critic = self.critic(x)
        cov_matrix = T.exp(self.log_std).expand_as(actor).diag_embed()
        
        dist = MultivariateNormal(actor, cov_matrix * temperature)
        
        return actor, critic.squeeze(-1), cov_matrix, dist
    
    def select_action(self, input, temperature: float = 1.):
        actor, critic_value, cov_matrix, dist = self.forward(input, temperature)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return {'action': action, 'value': critic_value, 'logprob': action_logprob, 'dist': dist, 'actor': actor, 'cov_matrix': cov_matrix}
    
    def calculate_loss(self, action, state, old_logprob, advantage, result):
        _, value, _, dist = self.forward(state)
        logprob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        
        r_t = T.exp(logprob - old_logprob)
        surr1 = advantage * r_t
        surr2 = advantage * r_t.clamp(1 - self.epsilon_, 1 + self.epsilon_)
        policy_loss = - T.min(surr1, surr2).mean()
        
        critic_loss = self.critic_loss_fn(value, result)
        
        loss = policy_loss + 0.5 * critic_loss + self.entropy_reg * entropy
        return loss
    
class PPO:
    def __init__(self, num_actions, hidden_dims: int = 16, num_layers: int = 2, final_activation: nn.Module | None = None, gamma_: float = 0.99, lambda_: float = 0.95, lr: float = 0.0003):
        self.policy = ActorCritic(num_actions=num_actions,  hidden_dims=hidden_dims, num_layers=num_layers, final_activation=final_activation)
        self.optimizer = Adam(self.policy.parameters(), lr=lr, weight_decay=0.)
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        
    def save_model(self):
        os.makedirs('outputs', exist_ok=True)
        T.save(self.policy.state_dict(), 'outputs//agent_state_dict.pth')
        T.save(self.optimizer.state_dict(), 'outputs//optimizer_state_dict.pth')
        
    def load_model(self):
        self.policy.load_state_dict(T.load('outputs//agent_state_dict.pth', weights_only=False))
        self.optimizer.load_state_dict(T.load('outputs//optimizer_state_dict.pth', weights_only=False))
        
    def select_action(self, input, temperature: float = 1.):
        return self.policy.select_action(input, temperature)
    
    def compute_advantage_and_results(self, reward, value, terminated, truncated):
        batch_size, timesteps = value.shape
        advantage = T.zeros(batch_size, timesteps-1)
        last_gae = 0
        
        for step in reversed(range(timesteps-1)):
            next_terminated_step = T.logical_not(terminated[:, step])
            truncated_step = truncated[:, step]
            next_truncated_step = T.logical_not(truncated_step.clone())
            delta = reward[:, step] + \
                self.gamma_ * next_terminated_step * next_truncated_step * value[:, step+1] +\
                    self.gamma_ * truncated_step * value[:, step] - \
                        value[:, step]
            last_gae = delta + last_gae * next_terminated_step * next_truncated_step * self.gamma_ * self.lambda_
            advantage[:, step] = last_gae
            
        result = advantage + value[:, :-1]
        return advantage, result
            
    
    def update(self, sample, ppo_epochs: int = 10):
        action, state, old_logprob, old_value, reward, terminated, truncated = sample
        action = action[:, :-1]
        state = state[:, :-1]
        old_logprob = old_logprob[:, :-1]
        
        advantage, result = self.compute_advantage_and_results(reward=reward, value=old_value, terminated=terminated, truncated=truncated)
        
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-6)
        
        for _ in range(ppo_epochs):
            self.optimizer.zero_grad()
            loss = self.policy.calculate_loss(action, state, old_logprob, advantage, result)
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        