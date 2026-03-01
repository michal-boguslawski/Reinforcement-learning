import torch as T
import torch.nn as nn

from ..models.models import Features
from ..torch_registry import ACTIVATION_FUNCTIONS


class SimpleCNN(nn.Module):
    def __init__(self, input_shape: tuple, num_features: int = 64, activation_fn: str = "relu"):
        super().__init__()
        self.input_shape = input_shape
        self.num_features = num_features
        self.out_features = num_features
        flattened_dim = 400
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]

        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, 8, 4, 0),
            self.activation(),
            nn.Conv2d(6, 16, 4, 4),
            self.activation(),
            nn.Flatten(),
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, num_features),
        )

    def forward(self, input_tensor: T.Tensor) -> Features:
        features = self.network(input_tensor)
        return Features(features=features)


class CNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple = (3, 96, 96),
        out_features: int = 64,
        dims: list = [32, 64, 128, 128],
        kernel_sizes: list = [5, 5, 3, 3],
        strides: list | None = [2, 2, 2, 1],
        paddings: list | None = None,
        activation_fn: str = "relu"
    ):
        super().__init__()
        self.input_shape = input_shape
        self.out_features = out_features
        self.dims = dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]

        self._build_network()

    def _build_network(self):
        dims = self.dims
        dims.insert(0, self.input_shape[0])

        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                nn.Conv2d(
                    dims[i],
                    dims[i+1],
                    self.kernel_sizes[i],
                    self.strides[i] if self.strides else 1,
                    self.paddings[i] if self.paddings else 0
                )
            )
            layers.append(self.activation())

        layers.append(nn.Flatten())
        self.network = nn.Sequential(*layers)
        
        self.fc = nn.Sequential(
            nn.LazyLinear(self.out_features),
            self.activation()
        )

    def forward(self, input_tensor: T.Tensor) -> Features:
        features = self.network(input_tensor)
        features = self.fc(features)
        return Features(features=features)
