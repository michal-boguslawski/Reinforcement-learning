import torch as T
import torch.nn as nn

from ..models.models import HeadOutput
from ..utils import activation_fns_dict


class ActorHead(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_features: int = 64,
        hidden_dim: int = 64,
        activation_fn: str = "tanh",
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fns_dict[activation_fn]

        self._build_network()

    def _build_network(self):
        
        self.actor = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            self.activation_fn(),
            nn.Linear(self.hidden_dim, self.num_actions),
        )

    def forward(self, features: T.Tensor) -> HeadOutput:
        actor_logits = self.actor(features)
        return HeadOutput(actor_logits=actor_logits)