from abc import ABC, abstractmethod
import torch as T
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.amp import GradScaler, autocast # type: ignore
from typing import Any, Generator, Dict

from .exploration.factory import get_exploration
from .callbacks.base import PolicyCallback
from memory.replay_buffer import ReplayBuffer
from models.models import ActionOutput, ActionSpaceType, OnPolicyMinibatch
from network.model import RLModel


class BasePolicy(ABC):
    network: RLModel
    optimizer: T.optim.Optimizer

    def __init__(
        self,
        network: RLModel,
        action_space_type: ActionSpaceType,
        exploration_method: Dict[str, Any],
        gamma_: float = 0.99,
        lambda_: float = 1.,
        buffer_size: int | None = None,
        device: T.device = T.device("cpu"),
        optimizer_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        *args,
        **kwargs
    ):
        self.network = network
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.buffer = ReplayBuffer(buffer_size=buffer_size, device=device)
        self.action_space_type = action_space_type
        self.device = device
        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 3e-4}
        self.max_grad_norm = 0.5
        self.loss_fn = loss_fn

        self._exploration_method = get_exploration(
            exploration_method_name=exploration_method["name"],
            exploration_kwargs=exploration_method.get("kwargs", {})
        )
        self._optimizer_setup()

        self._callbacks: list[PolicyCallback] = []

    def _optimizer_setup(self):
        from .factories import get_lr_scheduler
        self.optimizer = T.optim.Adam(self._build_param_groups(self.optimizer_kwargs))
        self.scheduler = get_lr_scheduler(optimizer=self.optimizer, **self.scheduler_kwargs)
        # self.scheduler = LinearLR(self.optimizer, start_factor=1., end_factor=1e-6, total_iters=5000)

    @abstractmethod
    def _calculate_loss(self, batch: OnPolicyMinibatch, temperature: float = 1) -> T.Tensor:
        pass

    @abstractmethod
    def _get_batch_for_training(self, *args, **kwargs) -> Dict[str, T.Tensor | None]:
        pass

    @abstractmethod
    def _generate_minibatches(self, minibatch_size: int, *args, **kwargs) -> Generator[OnPolicyMinibatch, None, None]:
        pass

    def train(self, minibatch_size: int, *args, **kwargs) -> None:
        batch = self._get_batch_for_training(*args, **kwargs)
        self._train_step(minibatch_size=minibatch_size, batch=batch, *args, **kwargs)
        if self.scheduler:
            self.scheduler.step()
            for i, val in enumerate(self.scheduler.get_last_lr()):
                self._emit_log(val, f"hyperparameters/lr_{i}")
        self._emit_flush()

    @property
    def has_critic(self) -> bool:
        return False

    def action(
        self,
        state: T.Tensor,
        core_state: T.Tensor | None = None,
        training: bool = True,
        temperature: float = 1.,
    ) -> ActionOutput:
        net = self.network
        assert isinstance(net, RLModel), f"Expected RLModel, got {type(net)}"

        with T.no_grad(), T.autocast("cuda", enabled=self.use_amp):
            output = net(input_tensor=state, core_state=core_state, temperature=temperature)
        
        action = self._exploration_method(
            logits = output.actor_logits,
            dist = output.dist,
            training = training,
            low = net.low,
            high = net.high,
        )

        log_prob = output.dist.log_prob(action)
        action_output = ActionOutput(
            action=action,
            logits=output.actor_logits,
            log_probs=log_prob,
            value=output.critic_value,
            dist=output.dist,
            core_state=output.core_state
        )
        return action_output

    def _train_step(self, minibatch_size: int, batch: Dict[str, T.Tensor | None], temperature: int = 1, *args, **kwargs) -> None:
        minibatch_generator = self._generate_minibatches(minibatch_size=minibatch_size, **batch)
        for minibatch in minibatch_generator:
            with T.autocast(device_type="cuda", enabled=self.use_amp):
                loss = self._calculate_loss(minibatch, temperature=temperature)
            self._emit_log(loss, "train/total_loss")
            self._backward(loss)

    def eval_mode(self) -> None:
        """Changing action network to eval mode"""
        self.network.eval()

    def train_mode(self) -> None:
        """Changing action network to train mode"""
        self.network.train()

    def _backward(self, loss: T.Tensor):
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

    def save_weights(self, folder_path: str):
        self.network.save_weights(folder_path)

    def load_weights(self, file_path: str, param_groups: list[str] | None = None):
        self.network.load_weights(file_path, param_groups)
    
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        self.buffer.push(item)

    def _build_param_groups(self, optimizer_kwargs: dict | None = None) -> list[dict]:
        optimizer_kwargs = optimizer_kwargs or {"lr": 3e-4}
        return [{"params": self.network.parameters(), "lr": optimizer_kwargs.get("lr", 3e-4)}]

    def add_callback(self, callback: PolicyCallback):
        self._callbacks.append(callback)

    def _emit_log(self, log: T.Tensor, name: str):
        for cb in self._callbacks:
            cb.on_log(log, name)

    def _emit_flush(self):
        for cb in self._callbacks:
            cb.flush()
