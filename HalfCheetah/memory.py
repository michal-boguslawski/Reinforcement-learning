from collections import deque
from random import sample, choices
import torch as T
import numpy as np


class Memory:
    def __init__(self, maxlen: int = 10000, backstep: int = 0):
        self.buffer = deque(maxlen=maxlen)
        self.state_buffer = deque(maxlen=maxlen+max(backstep, 1)-1)
        self.backstep = backstep
        
    def reset(self):
        self.buffer.clear()
        
    def push(self, state=None, item=None):
        if state is not None:
            self.state_buffer.append(state)
        if item is not None:
            self.buffer.append(item)
    
    def get(self, agg_type = T.cat, length: int = 1):
        items = [agg_type(column[-length:], dim=1) for column in zip(*self.buffer)]
        states = agg_type(list(self.state_buffer)[-(length+self.backstep):], dim=1)
        items.append(states)
        return items
    