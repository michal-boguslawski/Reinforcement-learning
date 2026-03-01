import torch as T
import torch.nn as nn

from ..models.models import HeadOutput
from ..utils import activation_fns_dict


class ActorCriticHead(nn.Module):
    def __init__(
        self,
        num_actions: int,
        in_features: int = 64,
        hidden_dim: int = 64,
        activation_fn: str = "tanh",
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_actions = num_actions
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fns_dict[activation_fn]

        self._build_network()

    def _build_network(self):
        
        self.actor = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            # nn.LayerNorm(self.hidden_dim),
            self.activation_fn(),
            nn.Linear(self.hidden_dim, self.num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            # nn.LayerNorm(self.hidden_dim),
            self.activation_fn(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: T.Tensor) -> HeadOutput:
        actor_logits = self.actor(features)
        critic_value = self.critic(features)
        return HeadOutput(actor_logits=actor_logits, critic_value=critic_value)
