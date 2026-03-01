import torch as T
from torch.distributions import Distribution


class EntropyMixin:
    @staticmethod
    def compute_entropy(dist: Distribution) -> T.Tensor:
        try:
            ent = dist.entropy()
        except NotImplementedError:
            ent = dist.base_dist.entropy()
        if ent.numel() == 0:
            raise ValueError("Empty entropy tensor")
        return ent.mean()
