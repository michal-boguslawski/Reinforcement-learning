import logging
import random
import torch as T
from torch.distributions import Distribution

from .base import BaseExploration


logger = logging.getLogger(__name__)


class EGreedyExploration(BaseExploration):
    def __init__(self, epsilon_start_: float = 1, epsilon_decay_: float = 1., decay_step_: int = 1_000, min_epsilon_: float = 0.02):
        self.epsilon_ = epsilon_start_
        self.epsilon_start_ = epsilon_start_
        self.epsilon_decay_ = epsilon_decay_
        self.decay_step_ = decay_step_
        self.min_epsilon_ = min_epsilon_
        self.k = 0

    def _counter(self):
        self.k += 1
        if self.k % self.decay_step_ == 0:
            self.epsilon_ *= self.epsilon_decay_
            self.epsilon_ = max(self.epsilon_, self.min_epsilon_)
            logger.debug(f"Current epsilon for egreedy {self.epsilon_}")

    def __call__(self, logits: T.Tensor, dist: Distribution, training: bool = True, *args, **kwargs) -> T.Tensor:
        if not training:
            return dist.sample()
        
        if self.epsilon_ > 0 and self.epsilon_ > random.random() and training:
            param_shape = getattr(dist, "param_shape")
            if param_shape is None:
                raise AttributeError("Distribution is wrong")
            batch_size, n = param_shape
            action = T.randint(0, n, (batch_size, ))
        else:
            action = logits.argmax(-1, keepdim=False)

        self._counter()

        return action
