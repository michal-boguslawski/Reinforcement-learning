import torch as T
import torch.nn as nn

from ..models.models import CoreOutput


class LSTMCore(nn.Module):
    def __init__(
        self,
        num_features: int = 64,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_features = num_features

        self._build_network()

    def _build_network(self):
        self.core = nn.LSTM(self.num_features, self.num_features, batch_first=True)

    def forward(self, features: T.Tensor, core_state: T.Tensor | None = None) -> CoreOutput:
        hx = None
        if core_state is not None:
            hx = (
                core_state[..., :self.num_features],
                core_state[..., self.num_features:]
            )
        core_out, hx = self.core(input=features, hx=hx)
        hx = T.cat(hx, dim=-1)
        return CoreOutput(core_out=core_out, core_state=hx)
