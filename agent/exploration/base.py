from abc import ABC, abstractmethod
import torch as T
from torch.distributions import Distribution


class BaseExploration(ABC):
    @abstractmethod
    def __call__(self, dist: Distribution, **kwargs) -> T.Tensor:
        pass
