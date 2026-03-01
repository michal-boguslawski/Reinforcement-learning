import torch as T
from typing import Protocol


class PolicyCallback(Protocol):
    def on_log(self, log: T.Tensor, name: str):
        pass

    def flush(self):
        pass
