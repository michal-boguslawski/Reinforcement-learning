import torch as T
from torch.distributions import Distribution, TransformedDistribution, TanhTransform, AffineTransform

from .base import DistributionTransform


class TanhAffineTransform(DistributionTransform):
    def __init__(self, low: T.Tensor, high: T.Tensor, *args, **kwargs):
        self.loc = (high + low) / 2
        self.scale = (high - low) / 2

    def apply(self, dist: Distribution) -> Distribution:
        return TransformedDistribution(
            dist,
            [
                TanhTransform(),
                AffineTransform(self.loc, self.scale),
            ],
        )
