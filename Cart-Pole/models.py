import torch as T # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore


class CartPoleDqn(nn.Module):
    def __init__(self, n_observations: int = 4, n_actions: int = 2, hidden_dim: int = 64, init_weight: float | None = None, **kwargs):
        super(CartPoleDqn, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.init_weight = init_weight
        self.init_norm = nn.Parameter(T.randn(n_observations, 2))
        self.weights = nn.Parameter(T.randn(n_observations, hidden_dim))
        self.hidden_dim = hidden_dim
        
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 4),
            nn.Flatten(),
            nn.GELU()
        )
        # self.encoder = nn.TransformerEncoderLayer(hidden_dim, 4, 4 * hidden_dim)
        
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions)
        )
        if init_weight:
            bias = self.model[-1].bias
            nn.init.constant_(bias, init_weight)
        
    def forward(self, state):
        input_shape = state.shape
        x = (state + self.init_norm[:, 0]) * self.init_norm[:, 1]
        x = x.unsqueeze(-1)
        x = x * self.weights
        x = F.relu(x)
        x = x.view((-1, self.n_observations, self.hidden_dim))
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # x = x[:, 0, :]
        logits = self.model(x)
        logits = logits.view(input_shape[:-1] + (-1, ))
        return logits
    