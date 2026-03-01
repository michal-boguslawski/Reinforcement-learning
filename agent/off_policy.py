from copy import deepcopy
import numpy as np
import random
from torch import nn
import torch as T
from torch.optim import Optimizer
from typing import Tuple, Generator, NamedTuple

from .base import BasePolicy
from memory.replay_buffer import ReplayBuffer
from .mixins import PolicyMixin
from models.models import Observation


class OffPolicy(PolicyMixin, BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, batch_size: int, minibatch_size: int = 64, **kwargs) -> np.floating | None:
        batch = self.buffer.sample(batch_size=batch_size)
        if not batch:
            return None
        minibatches = self._generate_minibatches(batch=batch, minibatch_size=minibatch_size)
        losses = []
        for minibatch in minibatches:
            loss = self.calculate_loss(minibatch)
            losses.append(loss)
        return np.mean(losses)

    def _generate_minibatches(
        self,
        batch: Observation,
        minibatch_size: int = 64
    ) -> Generator[Tuple[T.Tensor, ...], None, None]:
        batch = self._preprocess_batch(batch=batch)
        
        batch_size = batch.state.shape[0]
        minibatches = int(np.ceil(batch_size / minibatch_size))
        random_ids = np.random.permutation(batch_size)
        for minibatch_id in range(minibatches):
            start = minibatch_id * minibatch_size
            end = min((minibatch_id + 1) * minibatch_size, batch_size)
            minibatch_range = random_ids[start:end]
            yield (
                batch.state[minibatch_range],  # state
                batch.next_state[minibatch_range],  # next_state
                batch.action[minibatch_range],  # actions
                batch.reward[minibatch_range],  # rewards
                batch.done[minibatch_range],  # dones
            )


class DQNetworkPolicy(OffPolicy):
    def __init__(
        self,
        network: nn.Module,
        buffer_size: int,
        num_actions: int,
        optimizer: Optimizer,
        gamma_: float = 0.99,
        lambda_: float = 1,
        tau_: float = 0.005,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        *args,
        **kwargs
    ):
        super().__init__(num_actions=num_actions, gamma_=gamma_, lambda_=lambda_)
        self.policy_network = network
        self.target_network = deepcopy(network)
        self.buffer = ReplayBuffer(buffer_size)
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.tau_ = tau_
    
    @property
    def action_network(self) -> nn.Module:
        return self.policy_network

    def _soft_update(self):
        for target_param, source_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau_ * source_param.data + (1.0 - self.tau_) * target_param.data)

    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> float:
        states, next_states, actions, rewards, dones = batch
        
        outputs = self.policy_network(states)
        logits = outputs.logits

        with T.no_grad():
            next_state_outputs = self.target_network(next_states)
        next_state_values = next_state_outputs.logits.max(dim=-1).values

        q_values = logits.gather(dim=-1, index=actions).squeeze(-1)
        
        q_target = rewards + self.gamma_ * next_state_values * (1 - dones)
        
        loss = self.loss_fn(q_target, q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1)
        self.optimizer.step()
        
        self._soft_update()
        
        return loss.item()
