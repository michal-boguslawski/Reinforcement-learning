import torch as T
import torch.nn as nn
import torch.nn.functional as F


class SwimmerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers: int = 3):
        super(SwimmerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.GRUCell(hidden_dim, hidden_dim)
        )
        layers = []
        for i in range(num_layers):
            if i == 0:
                factor = 2
            else:
                factor = 1
            layers.append(nn.Linear(hidden_dim * factor, hidden_dim))
            layers.append(nn.ReLU())
        self.fcs = nn.Sequential(*layers)
        self.value_layer = nn.Linear(hidden_dim, 1)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(T.ones(output_dim))
        
    def forward(self, input, hx):
        x = self.fc1(input)
        hx_out = self.gru(hx)
        x = T.cat([x, hx_out], dim=-1)
        x = self.fcs(x)
        mean = self.mean_layer(x)
        std = T.diag_embed(T.exp(self.log_std))
        value = self.value_layer(x)
        return mean, std, value, hx_out
    