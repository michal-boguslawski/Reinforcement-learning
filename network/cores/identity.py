import torch as T
import torch.nn as nn

from ..models.models import CoreOutput


class IdentityCore(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        *args,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, features: T.Tensor, *args, **kwargs) -> CoreOutput:
        return CoreOutput(core_out=features)
