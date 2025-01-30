from collections import deque
from random import sample, choices
import torch as T # type: ignore
import numpy as np


class Memory:
    def __init__(self, maxlen: int = 10000, weight: float = 0.9999):
        self.buffer = deque(maxlen=maxlen)
        self.priority = deque(maxlen=maxlen)
        self.weights = weight ** (maxlen - np.arange(maxlen) - 1)
        
    def reset(self):
        self.buffer.clear()
        self.priority.clear()
        
    def push(self, item, priority = 1):
        self.buffer.append(item)
        self.priority.append(priority)
        
    def sample(self, sample_size: int = 16, agg_type = T.cat):
        weights = np.array(list(self.priority)) * self.weights[-len(self.priority):]
        sample_list = choices(self.buffer, weights=weights, k=sample_size)
        return [agg_type(column, dim=0) for column in zip(*sample_list)]