import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MountainCarModel(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 1, hidden_dim: int = 64):
        super(MountainCarModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.exp_dense = nn.Linear(input_dim, input_dim)
        self.layer = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        self.action_std = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
        
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, input):
        pom = self.exp_dense(input)
        pom = T.expm1(pom)
        x = T.cat([input, pom], dim=-1)
        x = self.layer(x)
        action_mean = self.action_mean(x)
        action_std = self.action_std(x)
        value = self.value(x)
        return action_mean, action_std.clamp(min=0.1), value
    
    
class ICM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 0.1
        self.beta = 0.2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.inverse = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.pi_logits = nn.Linear(hidden_dim, output_dim)
        
        self.dense1 = nn.Linear(hidden_dim+1, hidden_dim)
        self.phi_hat_new = nn.Linear(hidden_dim, hidden_dim)
        
        device = T.device('cpu')
        self.to(device)
        
        self.loss_fn = nn.HuberLoss()
        self.loss_fn_wo_reduction = nn.HuberLoss(reduction='none')
        
    def forward(self, states, next_states, actions):
        encoded_states = self.encoder(states)
        encoded_next_states = self.encoder(next_states)
        
        inverse = self.inverse(T.cat([encoded_states, encoded_next_states], dim=-1))
        pi_logits = self.pi_logits(inverse)
        
        forward_input = T.cat([encoded_states, actions], dim=-1)
        dense = self.dense1(forward_input)
        phi_hat_new = self.phi_hat_new(dense)

        return encoded_next_states, pi_logits, phi_hat_new
    
    def calc_loss(self, states, next_states, actions):
        encoded_next_states, pi_logits, phi_hat_new = self.forward(states, next_states, actions)
        
        L_I = (1 - self.beta) * self.loss_fn(pi_logits, actions)
        
        L_F = self.beta * self.loss_fn(phi_hat_new, encoded_next_states)
        
        intrinsic_reward = (self.alpha * 0.5 * self.loss_fn_wo_reduction(phi_hat_new, encoded_next_states)).sum(dim=-1)
        return intrinsic_reward.detach(), L_I, L_F
        