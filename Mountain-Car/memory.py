from collections import deque
from random import sample, choices
import torch as T
import numpy as np


class Memory:
    def __init__(self, maxlen: int = 10000, device='cpu'):
        self.buffer = deque(maxlen=maxlen)
        self.device = device
        
    def reset(self):
        self.buffer.clear()
        
    def push(self, item):
        self.buffer.append(item)
        
    def sample(self, sample_size: int = 16, agg_type = T.cat):
        weights = np.array(list(self.priority)) * self.weights[-len(self.priority):]
        sample_list = choices(self.buffer, weights=weights, k=sample_size)
        return [agg_type(column, dim=0) for column in zip(*sample_list)]
    
    def get(self, agg_type = T.cat):
        return [agg_type(column, dim=1).to(self.device) for column in zip(*self.buffer)]
    