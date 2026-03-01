from dataclasses import dataclass
import gymnasium as gym
from typing import NamedTuple, Literal, Tuple
import numpy as np
import torch as T
from torch.distributions import Distribution


ActionSpaceType = Literal["discrete", "continuous"]


@dataclass(slots=True)
class Observation:
    state: T.Tensor
    logits: T.Tensor
    action: T.Tensor
    reward: T.Tensor
    done: T.Tensor
    log_probs: T.Tensor
    dist: Distribution | None = None
    value: T.Tensor | None = None
    core_state: T.Tensor | None = None


@dataclass(slots=True)
class ActionOutput:
    action: T.Tensor
    logits: T.Tensor
    log_probs: T.Tensor
    value: T.Tensor | None = None
    dist: Distribution | None = None
    core_state: T.Tensor | None = None


@dataclass(slots=True)
class OnPolicyMinibatch:
    states: T.Tensor
    returns: T.Tensor
    actions: T.Tensor
    advantages: T.Tensor
    log_probs: T.Tensor
    state_values: T.Tensor
    core_states: T.Tensor | None = None


@dataclass(slots=True)
class EnvDetails:
    action_dim: int
    state_dim: Tuple[int, ...]
    action_space_type: ActionSpaceType
    action_low: T.Tensor | None
    action_high: T.Tensor | None
