import logging
import numpy as np
import torch as T


logger = logging.getLogger(__name__)


class RunningMeanStd:
    def __init__(self, eps=1e-4, device=T.device("cpu"), dtype=T.float32):
        self.mean = T.tensor(0.0, device=device, dtype=dtype)
        self.var = T.tensor(1.0, device=device, dtype=dtype)
        self.count = T.tensor(eps, device=device, dtype=dtype)

    def update(self, x: T.Tensor) -> None:
        x = x.to(self.mean.dtype)
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = T.tensor(x.numel(), device=x.device, dtype=x.dtype)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

        logger.debug(f"Running mean updated: mean={self.mean.item():.4f}, var={self.var.item():.4f}")

    def normalize(self, x: T.Tensor) -> T.Tensor:
        return x / (T.sqrt(self.var + 1e-8))


class RunningMeanStdEMA:
    def __init__(self, decay: float = 0.02, warmup_steps: int = 0, device=T.device("cpu"), dtype=T.float32):
        self.mean = T.zeros(1, device=device, dtype=dtype)
        self.var = T.ones(1, device=device, dtype=dtype)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.steps = 0
        self._init = False

    def update(self, x: T.Tensor):
        self.steps += 1
        x = x.to(self.mean.dtype)
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        
        if self._init:
            delta = batch_mean - self.mean
            self.mean = (1 - self.decay) * self.mean + self.decay * batch_mean
            self.var = (1 - self.decay) * (self.var + delta**2) + self.decay * batch_var
        else:
            self.mean = batch_mean
            self.var = batch_var
            self._init = True

        logger.debug(f"Running mean updated: mean={self.mean.item():.4f}, var={self.var.item():.4f}, steps {self.steps}")

    def normalize(self, x: T.Tensor) -> T.Tensor:
        if self.steps < self.warmup_steps:
            return x
        return x / (T.sqrt(self.var + 1e-8))


class RunningMeanStdFast:
    """
    Running mean and variance with:
      - EMA (exponential moving average)
      - Fast-start alpha for rapid initial adaptation
      - AMP / FP16/BF16 / GPU safe
    """
    def __init__(
        self,
        decay: float = 0.01,           # long-term EMA decay
        fast_start_alpha: float = 0.1,  # initial large alpha for fast adaptation
        fast_start_steps: int = 100,   # steps over which alpha decays to decay
        device: T.device = T.device("cpu"),
        dtype: T.dtype = T.float32
    ):
        self.mean = T.zeros(1, device=device, dtype=dtype)
        self.var = T.ones(1, device=device, dtype=dtype)
        self.decay = decay
        self.fast_start_alpha = fast_start_alpha
        self.fast_start_steps = fast_start_steps
        self.step = 0  # counter for adaptive alpha

    def update(self, x: T.Tensor):
        """
        Update running mean and variance with new batch x
        """
        x = x.to(device=self.mean.device)
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)

        self.step += 1

        # Compute adaptive alpha: starts at fast_start_alpha and decays to long-term decay
        if self.step < self.fast_start_steps:
            # Linear decay from fast_start_alpha -> decay
            alpha = self.fast_start_alpha - (
                (self.fast_start_alpha - self.decay) * (self.step / self.fast_start_steps)
            )
        else:
            alpha = self.decay

        # EMA update
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        self.var = (1 - alpha) * self.var + alpha * batch_var

        logger.debug(f"Running mean updated: mean={self.mean.item():.4f}, var={self.var.item():.4f}")

    def normalize(self, x: T.Tensor) -> T.Tensor:
        """
        Normalize x using running mean and variance
        """
        x = x.to(device=self.mean.device)
        return x / (T.sqrt(self.var + 1e-8))

    def state_dict(self) -> dict:
        """
        For saving/loading
        """
        return {
            "mean": self.mean,
            "var": self.var,
            "step": self.step,
        }

    def load_state_dict(self, state: dict):
        self.mean = state["mean"].to(self.mean.device, self.mean.dtype)
        self.var = state["var"].to(self.var.device, self.var.dtype)
        self.step = state["step"]
