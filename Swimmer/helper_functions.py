import gymnasium as gym
from gymnasium.wrappers import RecordVideo, Autoreset, NumpyToTorch, DtypeObservation
import torch as T
import os
import numpy as np


def make_env(env_id: str, dtype: T.dtype = np.float32, device: T.device = T.device('cpu'), name_prefix: str | None = None, **kwargs):
    if name_prefix is None:
        name_prefix = env_id
    env = gym.make(env_id, render_mode='rgb_array')
    env = Autoreset(env)
    env = DtypeObservation(env, dtype=dtype)
    env = NumpyToTorch(env, device=device)
    os.makedirs('save_videos', exist_ok=True)
    env = RecordVideo(env, video_folder='.//save_videos', fps=30, video_length=1000, name_prefix=env_id, step_trigger=lambda t: t % (1000 * 100) == 0)
    return env
