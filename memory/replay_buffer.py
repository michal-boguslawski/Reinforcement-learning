from collections import deque, namedtuple
from dataclasses import fields
import numpy as np
import random
import torch as T
from typing import NamedTuple, Any
from models.models import Observation

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int | None = None,
        device: T.device = T.device('cpu')
    ):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size) if buffer_size else deque()
        self._sample_method_choice = {
            1: self._sample_one_timestep
        }
        self.device = device
        # self.Observation = namedtuple('Observation', fieldnames)
        
    def __len__(self):
        return len(self.buffer)
        
    def push(self, item: dict[str, Any]):
        self.buffer.append(Observation(**item))
        
    def clear(self):
        self.buffer.clear()

    def _sample_one_timestep(self, batch_size: int, **kwargs) -> Observation:
        sample_list = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
        sample = tuple(T.stack(column, dim=0) for column in zip(*sample_list) if not any(v is None for v in column))
        sample = Observation(*sample)
        return sample

    def sample(self, batch_size: int, timesteps: int = 1, **kwargs) -> Observation | None:
        if len(self.buffer) < batch_size:
            return None
        # sample_method = self._sample_method_choice.get(timesteps, self._sample_multiple_timesteps)
        sample = self._sample_one_timestep(batch_size=batch_size)
        return sample
    
    def get_all(self) -> Observation:
        if len(self.buffer) == 0:
            raise ValueError("Cannot get_all from empty buffer")

        field_names = [f.name for f in fields(Observation)]
        columns = {name: [] for name in field_names}
        has_none = {name: False for name in field_names}

        try:
            # Single pass over buffer
            for item in self.buffer:
                for name in field_names:
                    value = getattr(item, name)
                    if value is None:
                        has_none[name] = True
                    columns[name].append(value)

            stacked = {}
            for name, values in columns.items():
                if has_none[name]:
                    continue
                stacked[name] = T.stack(values, dim=1)

            if not stacked:
                raise ValueError("No valid columns found in buffer data")

            sample = Observation(**stacked)

        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to stack buffer data: {e}")

        self.buffer.clear()
        return sample

    def get_trajectory(self):
        pass
