import numpy as np
import torch as T
from typing import Literal


def get_device(device: Literal["auto", "cpu", "cuda"] | T.device = T.device("cpu")):
    if isinstance(device, str) and device == "auto":
        device = T.device("cuda" if T.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = T.device(device)
    return device


def prepare_action_for_env(action: T.Tensor, action_space_type: str) -> np.ndarray | float | int:
    if action_space_type == "discrete" and action.numel() > 1:
        # For discrete actions, convert to scalar for single env or keep tensor for multiple envs
        env_action = action.detach().cpu().numpy()
        
    else:
        # For continuous actions, always convert to numpy array
        env_action = action.detach().cpu().numpy()

    return env_action
