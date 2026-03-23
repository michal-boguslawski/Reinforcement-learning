from torch.distributions import Distribution, Categorical
import torch as T

from .base import ActionBaseDistribution


class CategoricalDistribution(ActionBaseDistribution):
    def __init__(*args, **kwargs):
        pass

    def __call__(self, logits: T.Tensor, temperature: float = 1.0) -> Distribution:
        # Apply temperature
        logits = logits / temperature ** ( 1/2 )
        # Stabilize by subtracting max
        logits = logits - logits.max()
        return Categorical(logits=logits)
