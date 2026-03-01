import torch as T
import torch.nn as nn

from ..models.models import CoreOutput
from ..torch_registry import ACTIVATION_FUNCTIONS


class GRUCore(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 64,
        *args,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._build_network()

    def _build_network(self):
        
        self.core = nn.GRU(self.in_features, self.out_features, batch_first=False)
        

    def forward(self, features: T.Tensor, core_state: T.Tensor | None = None) -> CoreOutput:
        to_squeeze = False
        if features.dim() == 2 and (core_state is None or core_state.dim() == 3):
            features = features.unsqueeze(0)
            to_squeeze = True

        core_out, hx = self.core(input=features, hx=core_state)

        if to_squeeze:
            core_out = core_out.squeeze(0)
        return CoreOutput(core_out=core_out, core_state=hx)
