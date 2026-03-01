import torch as T
from torch.distributions import Distribution

from .base import OnPolicy
from ..mixins.entropy_mixin import EntropyMixin
from models.models import OnPolicyMinibatch
from network.heads.actor_critic import ActorCriticHead


class A2CPolicy(OnPolicy, EntropyMixin):
    def __init__(
        self,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_decay: float = 1.,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay

    @property
    def has_critic(self) -> bool:
        return True

    def _compute_policy_loss(self, dist: Distribution, actions: T.Tensor, advantages: T.Tensor) -> T.Tensor:
        # calculation policy loss
        log_probs = dist.log_prob(actions.squeeze(-1) if self.action_space_type == "discrete" else actions)
        log_probs = log_probs.sum(-1) if log_probs.ndim > 1 else log_probs

        policy_loss = -(log_probs * advantages.detach()).mean()
        self._emit_log(policy_loss, "train/policy_loss")
        return policy_loss

    def _compute_critic_loss(self, input: T.Tensor, target: T.Tensor):
        critic_loss = self.loss_fn(input.squeeze(-1), target.detach())
        self._emit_log(critic_loss, "train/critic_loss")
        return critic_loss

    def _calculate_loss(self, batch: OnPolicyMinibatch, temperature: float = 1.) -> T.Tensor:
        states, returns, actions, advantages, core_states = (
            batch.states,
            batch.returns,
            batch.actions,
            batch.advantages,
            batch.core_states
        )

        output = self.network(states, core_state=core_states, temperature=temperature)

        policy_loss = self._compute_policy_loss(output.dist, actions, advantages)
        critic_loss = self._compute_critic_loss(output.critic_value, returns)
        entropy = self.compute_entropy(output.dist)
        self._emit_log(entropy, "train/entropy")

        return policy_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

    def _build_param_groups(self, optimizer_kwargs: dict | None = None) -> list[dict]:
        optimizer_kwargs = optimizer_kwargs or {"lr": 3e-4}
        lr = optimizer_kwargs.get("lr")
        actor_lr = optimizer_kwargs.get("actor_lr") or lr
        critic_lr = optimizer_kwargs.get("critic_lr") or lr

        if not (
            isinstance(self.network.head, ActorCriticHead)
        ):
            raise NotImplementedError(
                "A2C requires the network to have an ActorCriticHead"
            )
        
        return [
            {"params": self.network.head.actor.parameters(), "lr": actor_lr},
            {"params": self.network.head.critic.parameters(), "lr": critic_lr},
            {"params": self.network.backbone.parameters(), "lr": critic_lr},
            {"params": self.network.core.parameters(), "lr": critic_lr},
            {"params": [self.network.log_std], "lr": actor_lr},
            {"params": [self.network.raw_scale_tril], "lr": actor_lr},
        ]
