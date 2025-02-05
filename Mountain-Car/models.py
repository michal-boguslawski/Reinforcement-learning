import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MountainCarModel(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 1, hidden_dim: int = 64):
        super(MountainCarModel, self).__init__()
        self.norm = nn.Parameter(T.randn(1))
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        x = self.norm * input
        x = self.layer(x)
        action_mean = self.action_mean(x)
        action_std = self.action_std(x)
        value = self.value(x)
        return action_mean, action_std.clamp(min=0.1), value
    