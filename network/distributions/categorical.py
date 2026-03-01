from torch.distributions import Distribution, Categorical
import torch as T

from .base import ActionBaseDistribution


class CategoricalDistribution(ActionBaseDistribution):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, logits: T.Tensor, temperature: float = 1.0) -> Distribution:
        return Categorical(logits=logits / temperature)
