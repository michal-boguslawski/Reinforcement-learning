import torch as T
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import HuberLoss
import os
import numpy as np

T.set_printoptions(precision=4, sci_mode=False, linewidth=200)
np.printoptions(precision=4, suppress=True)

def create_disc_table(batch_size, timesteps, v):
    disc_table = T.arange(timesteps).unsqueeze(1) - T.arange(timesteps)
    disc_table = v ** disc_table
    disc_table = disc_table.tril()
    disc_table = disc_table.unsqueeze(0).expand(batch_size, -1, -1)
    return disc_table

class GRUExtractor(nn.Module):
    def __init__(self, input_dim, gru_dims, activation: nn.Module = nn.Tanh):
        super(GRUExtractor, self).__init__()
        self.gru_dims = gru_dims
        self.pre = nn.Sequential(
            nn.Linear(input_dim, gru_dims),
            nn.Tanh()
        )
        self.gru = nn.GRU(gru_dims, gru_dims, num_layers=1, batch_first=True)
        self.after = nn.Sequential(
            nn.Linear(gru_dims, gru_dims),
            nn.Tanh()
        )
        self.activation = activation()
    
    def forward(self, input):
        input_shape = input.shape
        x = input.reshape((-1, ) + input_shape[-2:])
        x = self.pre(x)
        x, _ = self.gru(x)
        x = self.activation(x)
        x = x[..., -1, :]
        x = self.after(x)
        x = x.reshape(input_shape[0], -1, self.gru_dims).squeeze(1)
        return x

class ActorCritic(nn.Module):
    def __init__(self, 
                 num_actions: int, 
                 input_dim: int,
                 hidden_dims: int = 16, 
                 gru_dims: int = 16,
                 if_rnn: bool = False,
                 num_layers: int = 2, 
                 epsilon_: float = 0.2, 
                 final_activation: nn.Module | None = None, 
                 entropy_reg: float = 0.):
        super(ActorCritic, self).__init__()
        
        self.if_rnn = if_rnn
        self.input_dim = input_dim
        self.gru_dims = gru_dims
        self.num_actions = num_actions
        if if_rnn:
            self.extractor = GRUExtractor(input_dim, gru_dims)
        else:
        
            self.mlp_dims = 2 * hidden_dims
            layers = []
            for _ in range(1):
                layers.append(nn.Linear(input_dim, self.mlp_dims))
                layers.append(nn.Tanh())
            self.extractor = nn.Sequential(*layers)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LazyLinear(hidden_dims))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dims, num_actions))
        # if final_activation:
        #     layers.append(final_activation)
        self.actor = nn.Sequential(*layers)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LazyLinear(hidden_dims))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dims, 1))
        self.critic = nn.Sequential(*layers)
        
        self.log_std = nn.Parameter(T.zeros(num_actions))
        self.scale_tril = nn.Parameter(T.zeros(num_actions, num_actions))
        self.final_activation = final_activation
        
        self.epsilon_ = epsilon_
        self.entropy_reg = entropy_reg
        self.critic_loss_fn = HuberLoss()
        
    def forward(self, input, temperature: float = 1.):
        x = self.extractor(input)
        
        actor = self.actor(x)
        critic = self.critic(x)
        cov_matrix = T.exp(self.log_std).diag_embed()
        scale_tril = cov_matrix + self.scale_tril.tril() * (1 - T.eye(self.num_actions))
        
        # scale_tril = self.scale_tril @ self.scale_tril.T
        
        dist = MultivariateNormal(
            actor, 
            # covariance_matrix=cov_matrix*temperature
            scale_tril=scale_tril*temperature
            )
        
        return actor, critic.squeeze(-1), dist.covariance_matrix, dist
    
    def select_action(self, input, temperature: float = 1.):
        actor, critic_value, cov_matrix, dist = self.forward(input, temperature)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return {
            'action': action, 
            'value': critic_value, 
            'logprob': action_logprob, 
            'dist': dist, 
            'actor': actor, 
            'cov_matrix': cov_matrix
            }
    
    def calculate_loss(self, action, state, old_logprob, advantage, result, factor: float = 1):
        _, value, _, dist = self.forward(state)
        logprob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        
        r_t = T.exp(logprob - old_logprob)
        surr1 = advantage * r_t
        surr2 = advantage * r_t.clamp(1 - self.epsilon_, 1 + self.epsilon_)
        policy_loss = - T.min(surr1, surr2).mean()
        
        critic_loss = self.critic_loss_fn(value, result)
        
        loss = policy_loss + 0.5 * critic_loss + (2 * factor - 1) * self.entropy_reg * entropy
        return loss, policy_loss, critic_loss, entropy
    
class PPO:
    def __init__(self, 
                 num_actions: int, 
                 input_dim: int,
                 hidden_dims: int = 16, 
                 if_rnn: bool = False,
                 gru_dims: int = 16, 
                 num_layers: int = 2, 
                 final_activation: nn.Module | None = None, 
                 gamma_: float = 0.99, 
                 lambda_: float = 0.95, 
                 lr: float = 0.0003,
                 backstep: int = 3,
                 entropy_reg: float = 0,
                 device: T.device = T.device('cpu')):
        self.num_actions = num_actions
        self.policy = ActorCritic(num_actions=num_actions, 
                                  input_dim=input_dim, 
                                  hidden_dims=hidden_dims, 
                                  gru_dims=gru_dims, 
                                  if_rnn=if_rnn, 
                                  num_layers=num_layers, 
                                  entropy_reg=entropy_reg,
                                  final_activation=final_activation).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr, weight_decay=0.)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20000, eta_min=1e-8)
        self.device = device
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.backstep = backstep
        
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
        batch_size, timesteps = reward.shape
        not_terminated = T.logical_not(terminated)
        not_truncated = T.logical_not(truncated)
        
        delta = reward + self.gamma_ * not_terminated * not_truncated * value[:, 1:] +\
            self.gamma_ * truncated * value[:, :-1] -\
                value[:, :-1]
        
        dones = T.logical_or(terminated, truncated)
        done_indices = T.nonzero(dones)
        disc_table = create_disc_table(batch_size, timesteps, self.gamma_ * self.lambda_).to(self.device)
        for i, idx in done_indices:
            disc_table[i, (idx+1):, :(idx+1)] = 0
            
        advantage = T.einsum('ni, nij -> nj', delta, disc_table)
        
        result = advantage + value[:, :-1]
        return advantage, result
            
    
    def update(self, sample, ppo_epochs: int = 10, backstep: int = 0, factor: float = 1., mini_batchsize: int = 64):
        action, old_logprob, old_value, reward, terminated, truncated, state = sample
        action = action[:, :-1]
        old_logprob = old_logprob[:, :-1]
        state = state[:, :-1]
        reward = reward[:, :-1]
        terminated = terminated[:, :-1]
        truncated = truncated[:, :-1]
        if backstep > 1:
            state = state.unfold(1, backstep, 1)
            state = state.transpose(-1, -2)
        
        advantage, result = self.compute_advantage_and_results(reward=reward, value=old_value, terminated=terminated, truncated=truncated)
        
        # advantage = (advantage - advantage.mean())/(advantage.std() + 1e-6)
        
        losses = []
        
        pom = np.ceil(action.shape[1]/mini_batchsize, casting='unsafe', dtype=np.int64)
        size = action.shape[0] * pom
        
        for _ in range(ppo_epochs):
            for idx in np.random.permutation(size):
                i, k = np.ceil(idx/pom, casting='unsafe', dtype=np.int64), idx % pom
                
                batch_action = action[i, (k * mini_batchsize):(k * mini_batchsize + mini_batchsize)]
                batch_state = state[i, (k * mini_batchsize):(k * mini_batchsize + mini_batchsize)]
                batch_old_logprob = old_logprob[i, (k * mini_batchsize):(k * mini_batchsize + mini_batchsize)]
                batch_advantage = advantage[i, (k * mini_batchsize):(k * mini_batchsize + mini_batchsize)]
                batch_advantage = (batch_advantage - batch_advantage.mean())/(batch_advantage.std() + 1e-6)
                batch_result = result[i, (k * mini_batchsize):(k * mini_batchsize + mini_batchsize)]
                
                self.optimizer.zero_grad()
                loss = self.policy.calculate_loss(batch_action, batch_state, batch_old_logprob, batch_advantage, batch_result, factor=factor)
                loss[0].backward()
                T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                losses.append([x.item() for x in loss])
            # self.scheduler.step()
        losses = [np.mean(col) for col in zip(*losses)]
        print(f"Total loss {losses[0]:.4f}, policy loss {losses[1]:.4f}, critic loss {losses[2]:.4f}, entropy {losses[3]:.4f}")