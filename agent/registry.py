# Policies
from .on_policy.sarsa import SarsaPolicy
from .on_policy.a2c import A2CPolicy
from .on_policy.ppo import PPOPolicy

#Schedulers
from .schedulers.entropy import LinearSchedule
from torch.optim.lr_scheduler import LinearLR, OneCycleLR, ExponentialLR, CosineAnnealingLR


POLICIES = {
    "a2c": A2CPolicy,
    "ppo": PPOPolicy,
    "sarsa": SarsaPolicy,
}


SCHEDULERS = {
    "linear_entropy": LinearSchedule,
    "linear_lr": LinearLR,
    "one_cycle_lr": OneCycleLR,
    "exponential_lr": ExponentialLR,
    "cosine_annealing_lr": CosineAnnealingLR
}
