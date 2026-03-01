import os
from pathlib import Path
import torch as T

from .load_model import load_model
from agent.on_policy import PPOPolicy
from models.models import OnPolicyMinibatch


model = load_model()

agent = PPOPolicy(
    network=model,
    action_space_type="continuous",
    exploration_method={"name": "distribution"},
    advantage_normalize="batch",
    gamma_=0.99,
    lambda_=0.95,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    clip_range=0.2,
    lr=3e-4,
    n_epochs=10,
    device=T.device("cpu")
)

batch = OnPolicyMinibatch(
    states=T.load("/app/rl_agents/tests/unit/network/data/states.pt"),
    returns=T.load("/app/rl_agents/tests/unit/network/data/returns.pt"),
    actions=T.load("/app/rl_agents/tests/unit/network/data/actions.pt"),
    advantages=T.load("/app/rl_agents/tests/unit/network/data/advantages.pt"),
    log_probs=T.load("/app/rl_agents/tests/unit/network/data/old_log_probs.pt")
)

expected_loss = (0.12188223004341125, -2.551823854446411e-07, 0.34525254368782043, 5.0743794441223145)

loss = agent.calculate_loss(batch=batch)

assert loss == expected_loss, f"Expected {expected_loss}, got {loss}"
print("Test passed")
