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
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus()
        )
        self.gru = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.GRUCell(hidden_dim, hidden_dim)
        )
        layers = []
        for i in range(num_layers):
            if i == 0:
                factor = 3
            else:
                factor = 1
            layers.append(nn.Linear(hidden_dim * factor, hidden_dim))
            layers.append(nn.ReLU())
        self.fcs = nn.Sequential(*layers)
        self.value_layer = nn.Linear(hidden_dim, 1)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(T.ones(output_dim))
        
    def forward(self, input, hx):
        x1 = self.fc1(input)
        x2 = self.fc2(input)
        hx = hx.view(-1, self.hidden_dim)
        hx_out = self.gru(hx)
        hx_out = hx_out.view(len(input), -1, self.hidden_dim).squeeze(1)
        x = T.cat([x1, x2, hx_out], dim=-1)
        x = self.fcs(x)
        mean = self.mean_layer(x)
        std = T.exp(self.log_std).expand_as(mean)
        std = T.diag_embed(std)
        value = self.value_layer(x)
        return mean, std, value, hx_out
    