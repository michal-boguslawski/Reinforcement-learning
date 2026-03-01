import torch as T
import torch.nn as nn

from ..models.models import Features
from ..torch_registry import ACTIVATION_FUNCTIONS


class MLPNetwork(nn.Module):
    def __init__(
        self,
        input_shape: int | tuple,
        num_features: int = 64,
        hidden_dims: int = 64,
        num_layers: int = 2,
        activation_fn: str = "tanh"
    ):
        super().__init__()
        self.input_shape = input_shape if isinstance(input_shape, int) else input_shape[0]
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_features = num_features
        self.activation_fn = ACTIVATION_FUNCTIONS[activation_fn]

        self._build_network()

    def _build_network(self):
        modules = []
        for i in range(self.num_layers):
            modules.append(
                nn.Linear(
                    in_features=self.input_shape if i == 0 else self.hidden_dims,
                    out_features=self.num_features if i == (self.num_layers - 1) else self.hidden_dims
                )
            )
            modules.append(self.activation_fn())
        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor: T.Tensor) -> Features:
        features = self.network(input_tensor)
        return Features(features=features)
