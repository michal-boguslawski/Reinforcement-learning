from collections import deque
from random import sample, choices
import torch as T
import numpy as np


class Memory:
    def __init__(self, maxlen: int = 10000):
        self.buffer = deque(maxlen=maxlen)
        
    def reset(self):
        self.buffer.clear()
        
    def push(self, item):
        self.buffer.append(item)
    
    def get(self, agg_type = T.cat, length: int = 1):
        return [agg_type(column, dim=1)[-length:] for column in zip(*self.buffer)]
    