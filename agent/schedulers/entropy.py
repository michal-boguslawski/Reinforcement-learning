import logging


logger = logging.getLogger(__name__)


class LinearSchedule:
    def __init__(self, max_entropy: float, total_steps: int, min_entropy: float | None = None):
        self.max_entropy = max_entropy
        self.min_entropy = max_entropy if min_entropy is None else min_entropy
        self.total_steps = total_steps
        self._curr_step = 0

    def __call__(self):
        frac = min(self._curr_step / self.total_steps, 1.0)
        return self.max_entropy + frac * (self.min_entropy - self.max_entropy)

    def step(self):
        self._curr_step += 1
        logger.debug({"hyperparameters/entropy_coef": self.__call__()})

    def reset(self):
        self._curr_step = 0
