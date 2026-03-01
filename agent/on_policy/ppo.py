import torch as T
from torch.distributions import Distribution
from typing import Dict

from .base import OnPolicy
from ..mixins.entropy_mixin import EntropyMixin
from models.models import OnPolicyMinibatch
from network.heads.actor_critic import ActorCriticHead


class PPOPolicy(OnPolicy, EntropyMixin):
    def __init__(
        self,
        value_loss_coef: float = 0.5,
        entropy_kwargs: dict = {},
        num_epochs: int = 10,
        clip_epsilon: float = 0.2,
        use_value_clipping: str | None = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.value_loss_coef = value_loss_coef
        self.entropy_kwargs = entropy_kwargs
        self.num_epochs = num_epochs
        self.clip_epsilon = clip_epsilon
        self.use_value_clipping = use_value_clipping

        self.loss_fn = T.nn.HuberLoss(reduction="none")
        self._entropy_setup()

    def _entropy_setup(self):
        from ..factories import get_scheduler
        self.entropy_coef = get_scheduler(**self.entropy_kwargs)

    @property
    def has_critic(self) -> bool:
        return True

    def _train_step(self, minibatch_size: int, batch: Dict[str, T.Tensor | None], *args, **kwargs):
        for _ in range(self.num_epochs):
            super()._train_step(minibatch_size, batch, *args, **kwargs)
        self.entropy_coef.step()

    def _compute_policy_loss(
        self,
        dist: Distribution,
        actions: T.Tensor,
        old_log_probs: T.Tensor,
        advantages: T.Tensor
    ):
        log_probs = dist.log_prob(actions.squeeze(-1) if self.action_space_type == "discrete" else actions)
        log_ratio = log_probs - old_log_probs.detach()
        log_ratio = T.clamp(log_ratio, -10, 10)
        r_t = T.exp(log_ratio)
        r_t = r_t.sum(-1) if r_t.ndim > 1 else r_t

        policy_loss = -(T.min(
            r_t * advantages,
            T.clamp(r_t, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        )).mean()
        self._emit_log(policy_loss, "train/policy_loss")

        clip_frac = ((r_t.detach() - 1).abs() > self.clip_epsilon).float().mean()
        self._emit_log(clip_frac, "train/clip_fraction")

        approx_kl = (old_log_probs.detach() - log_probs.detach()).mean()
        self._emit_log(approx_kl, "train/kl_div")
        return policy_loss

    def _compute_critic_loss(
        self,
        values: T.Tensor,
        old_values: T.Tensor,
        returns: T.Tensor
    ):
        if self.use_value_clipping == "absolute":

            values_clipped = old_values + (values - old_values).clamp(-self.clip_epsilon, self.clip_epsilon)
            loss_unclipped = self.loss_fn(values, returns)
            loss_clipped = self.loss_fn(values_clipped, returns)
            critic_loss = T.max(loss_unclipped, loss_clipped).mean()

            clip_frac = (loss_clipped > loss_unclipped).float().mean()
            self._emit_log(clip_frac, "train/value_clip_fraction")

        elif self.use_value_clipping == "relative":
            values_clipped = old_values + (values - old_values).clamp(
                -self.clip_epsilon * old_values.abs().clamp(min=1),
                self.clip_epsilon * old_values.abs().clamp(min=1)
            )
            loss_unclipped = self.loss_fn(values, returns)
            loss_clipped = self.loss_fn(values_clipped, returns)

            critic_loss = T.max(loss_unclipped, loss_clipped).mean()

            clip_frac = (loss_clipped > loss_unclipped).float().mean()
            self._emit_log(clip_frac, "train/value_clip_fraction")

        else:
            critic_loss = self.loss_fn(values, returns).mean()

        self._emit_log(critic_loss, "train/critic_loss")
        return critic_loss

    def _calculate_loss(self, batch: OnPolicyMinibatch, temperature: float = 1.) -> T.Tensor:
        states, returns, actions, advantages, old_log_probs, old_values, core_states = (
            batch.states,
            batch.returns,
            batch.actions,
            batch.advantages,
            batch.log_probs,
            batch.state_values,
            batch.core_states
        )
        output = self.network(states, core_state=core_states, temperature=temperature)

        policy_loss = self._compute_policy_loss(
            output.dist,
            actions,
            old_log_probs,
            advantages.detach()
        )
        critic_loss = self._compute_critic_loss(
            output.critic_value.squeeze(-1),
            old_values.detach(),
            returns.detach()
        )
        entropy = self.compute_entropy(output.dist)
        self._emit_log(entropy, "train/entropy")
        return policy_loss + self.value_loss_coef * critic_loss - self.entropy_coef() * entropy

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
