import torch as T
from models import MountainCarModel
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn import MSELoss, HuberLoss
from torch.distributions.normal import Normal


def get_discounted_table(n, v):
    tab = T.arange(n).unsqueeze(1) - T.arange(n)
    tab = v ** tab
    return tab.tril()

       
class ACAgent:
    def __init__(self, input_dim: int = 2, output_dim: int = 1, timesteps: int = 5, hidden_dim: int = 64, device = 'cpu'):
        self.lambda_ = 0.9
        self.gamma_ = 0.95
        self.agent = MountainCarModel(
            input_dim, 
            output_dim, 
            hidden_dim
            ).to(device)
        self.device = device
        self.loss_fn = HuberLoss(reduction='none')
        self.discounted_table = get_discounted_table(timesteps, self.gamma_ * self.lambda_).to(device)
        self.upper_triangle = T.ones(timesteps, timesteps).triu().to(device)
        
        
    def action(self, state, distribution: str = "normal", temperature: float = 1, min_std: float = 0.):
        state = state.to(self.device)
        action_mean, action_std, state_value = self.agent(state)
        dist = Normal(action_mean, 
                      (action_std * temperature).clip(min=min_std)
                      )
        action = dist.rsample() #rsample for gradients
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return T.tanh(action), log_prob, state_value, entropy
    
    def train(self, sample, entropy_reg, ICM):
        states, state_values, actions, entropy, log_probs, rewards, dones = sample
        
        next_states = states[:, 1:].clone().detach().squeeze(-1)
        states = states[:, :-1].detach().squeeze(-1)
        actions = actions[:, :-1]
        intrinsic_reward, L_I, L_F = ICM.calc_loss(states, next_states, actions)
    
        next_state_values = state_values[:, 1:].clone().detach().squeeze(-1)
        state_values = state_values[:, :-1].squeeze(-1)
        
        dones = dones[:, :-1].to(T.float)
        entropy = entropy[:, :-1].squeeze(-1)
        log_probs = log_probs[:, :-1].squeeze(-1)
        rewards = rewards[:, :-1] + 100. * intrinsic_reward
        following_dones = T.matmul(dones, self.upper_triangle)
        
        td_errors = rewards + self.gamma_ * next_state_values * (1 - dones) - state_values
        td_errors = td_errors * (1 - following_dones + dones)
        
        discounted_td_errors = T.matmul(td_errors, self.discounted_table)
        
        policy_loss = - (td_errors.detach() * log_probs)
        
        # critic_loss = T.where(discounted_td_errors.abs() > 1, discounted_td_errors.pow(2), discounted_td_errors.abs())
        critic_loss = self.loss_fn(discounted_td_errors, T.zeros_like(discounted_td_errors))
        # critic_loss = critic_loss.mean()
        
        loss = policy_loss + 0.5 * critic_loss - entropy_reg * entropy
        loss = loss.mean()
        return (loss, policy_loss.mean(), critic_loss.mean(), entropy.mean()), L_I, L_F
