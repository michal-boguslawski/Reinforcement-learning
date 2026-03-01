import torch as T
from torch import nn

from .base import BasePolicy
from .registry import POLICIES, SCHEDULERS
from .callbacks.train_logger import TrainPolicyLogger


def get_policy(
    policy_type: str,
    network: nn.Module,
    action_space_type: str,
    policy_kwargs: dict,
    device: T.device = T.device("cpu"),
    verbose: int = 1,
) -> BasePolicy:
    agent = POLICIES.get(policy_type)
    if agent is None:
        raise ValueError(f"Agent {policy_type} does not exist")

    policy: BasePolicy = agent(
        network=network,
        action_space_type=action_space_type,
        device=device,
        **policy_kwargs
    )

    if verbose == 1:
        policy.add_callback(TrainPolicyLogger())

    return policy


def get_scheduler(
    scheduler_type: str,
    *args,
    **kwargs
):
    scheduler = SCHEDULERS.get(scheduler_type)
    if scheduler is None:
        raise ValueError(f"Scheduler {scheduler_type} does not exist")
    return scheduler(**kwargs)


def get_lr_scheduler(
    optimizer,
    scheduler_type: str | None = None,
    *args,
    **kwargs
):
    if scheduler_type is None:
        return None
    scheduler = SCHEDULERS.get(scheduler_type)
    if scheduler is None:
        raise ValueError(f"Scheduler {scheduler_type} does not exist")
    return scheduler(optimizer=optimizer, **kwargs)
