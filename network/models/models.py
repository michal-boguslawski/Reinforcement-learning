from dataclasses import dataclass
import torch as T
from torch.distributions import Distribution


@dataclass(slots=True)
class Features:
    features: T.Tensor


@dataclass(slots=True)
class HeadOutput:
    actor_logits: T.Tensor
    critic_value: T.Tensor | None = None


@dataclass(slots=True)
class ModelOutput:
    actor_logits: T.Tensor
    dist: Distribution
    critic_value: T.Tensor | None = None
    core_state: T.Tensor | None = None


@dataclass(slots=True)
class CoreOutput:
    core_out: T.Tensor
    core_state: T.Tensor | None = None
