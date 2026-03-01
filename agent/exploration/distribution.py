from .base import BaseExploration
import torch as T
from torch.distributions import Distribution


class DistributionExploration(BaseExploration):
    def __call__(self, dist: Distribution, low: T.Tensor | None = None, high: T.Tensor | None = None, *args, **kwargs) -> T.Tensor:
        action = dist.sample()
        if low is not None and high is not None:
            action = action.clamp(low + 1e-5, high - 1e-5)

        return action
