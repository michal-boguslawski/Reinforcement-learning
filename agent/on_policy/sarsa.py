import torch as T

from .base import OnPolicy
from models.models import OnPolicyMinibatch


class SarsaPolicy(OnPolicy):
    def _calculate_loss(self, batch: OnPolicyMinibatch, temperature: float = 1) -> T.Tensor:
        states, results, actions, core_states = (
            batch.states,
            batch.returns,
            batch.actions,
            batch.core_states
        )
        
        output = self.network(states, core_state=core_states, temperature=temperature)
        
        q_values = output.actor_logits.gather(dim=-1, index=actions).squeeze(-1)
        loss = self.loss_fn(q_values, results)
        return loss
