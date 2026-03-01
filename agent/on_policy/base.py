import logging
import numpy as np
import random
import torch as T
from typing import Dict, Generator, Tuple

from ..base import BasePolicy
from ..utils.preprocessing import preprocess_batch
from ..utils.running_mean import RunningMeanStdEMA
from models.models import ActionSpaceType, Observation, OnPolicyMinibatch
from utils.utils import compute_advantage_and_results


logger = logging.getLogger(__name__)


class OnPolicy(BasePolicy):
    action_space_type: ActionSpaceType

    def __init__(self, advantage_normalize: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.advantage_normalize = advantage_normalize
    
    def _extract_state_values(self, batch: Observation) -> Tuple[T.Tensor, T.Tensor]:
        if self.has_critic:
            if not isinstance(batch.value, T.Tensor):
                raise ValueError("Value tensor is required for critic-based agents")

            state_values = batch.value.squeeze(-1)

        else:
            state_values = batch.logits.gather(dim=-1, index=batch.action).squeeze(-1)

        next_state_values = state_values[:, 1:]
        return state_values[:, :-1], next_state_values

    def _get_batch_for_training(self, *args, **kwargs) -> Dict[str, T.Tensor | None]:
        raw_batch = self.buffer.get_all()
        preprocessed_batch = preprocess_batch(raw_batch, self.action_space_type)
        state_values, next_state_values = self._extract_state_values(preprocessed_batch)

        returns, advantages = compute_advantage_and_results(
                rewards=preprocessed_batch.reward[:, :-1],
                dones=preprocessed_batch.done[:, :-1],
                state_values=state_values,
                next_state_values=next_state_values,
                gamma_=self.gamma_,
                lambda_=self.lambda_
        )

        self._emit_log(returns, "stats/mean_returns")
        self._emit_log(advantages, "stats/mean_advantages")

        if self.advantage_normalize == "global":
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        
        return {
            "returns": returns.flatten(0, 1).contiguous(),
            "advantages": advantages.flatten(0, 1).contiguous(),
            "states": preprocessed_batch.state[:, :-1].flatten(0, 1).contiguous(),
            "actions": preprocessed_batch.action[:, :-1].flatten(0, 1).contiguous(),
            "log_probs": preprocessed_batch.log_probs[:, :-1].flatten(0, 1).contiguous(),
            "state_values": 
                state_values.flatten(0, 1).contiguous(),
            "core_states": 
                preprocessed_batch.core_state[:, :-1].transpose(1, 2).flatten(1, 2).contiguous()
                if preprocessed_batch.core_state is not None else None,
        }

    def _generate_minibatches(
        self,
        returns: T.Tensor,
        advantages: T.Tensor,
        states: T.Tensor,
        actions: T.Tensor,
        log_probs: T.Tensor,
        state_values: T.Tensor,
        core_states: T.Tensor | None = None,
        minibatch_size: int = 64
    ) -> Generator[OnPolicyMinibatch, None, None]:
        batch_size = int(np.prod(returns.shape))
        indices = T.arange(0, batch_size, device=self.device)
        minibatch_size = min(batch_size, minibatch_size)
        starts = list(range(0, batch_size, minibatch_size))
        random.shuffle(starts)

        for start in starts:
            end = min(start + minibatch_size, batch_size - 1)
            mb_idx = indices[start:end]
            batch_advantages = advantages[mb_idx]

            if self.advantage_normalize == "batch":
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-6)

            yield OnPolicyMinibatch(
                states=states[mb_idx],
                returns=returns[mb_idx],
                actions=actions[mb_idx],
                advantages=batch_advantages,
                log_probs=log_probs[mb_idx],
                state_values=state_values[mb_idx],
                core_states=core_states[:, mb_idx] if core_states is not None else None,
            )
