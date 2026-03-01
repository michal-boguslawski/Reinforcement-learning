import inspect
import numpy as np

from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.vector import VectorEnv
from models.models import EnvDetails


def get_env_vec_details(env: VectorEnv):
    action_space = getattr(env, "action_space")
    observation_space = getattr(env, "observation_space")
    
    action_space_type = "discrete" if isinstance(env.action_space, Discrete) \
        or isinstance(env.action_space, MultiDiscrete) else "continuous"

    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    elif isinstance (env.action_space, MultiDiscrete):
        action_dim = env.action_space.nvec[0]
    else:
        action_dim = action_space.shape[-1]

    low = getattr(action_space, "low", None)
    if low is not None:
        low = low[0]

    high = getattr(action_space, "high", None)
    if high is not None:
        high = high[0]
    
    return EnvDetails(
        action_dim=int(action_dim),
        state_dim=observation_space.shape[1:],
        action_space_type=action_space_type,
        action_low=low,
        action_high=high,
    )


def clean_kwargs(func, kwargs: dict):
    """
    Return a new dict containing only keys that appear
    in the function's signature.
    """
    sig = inspect.signature(func)
    valid = set(sig.parameters.keys())

    return {k: v for k, v in kwargs.items() if k in valid}


def image_border(
    obs: np.ndarray,
    y_range=(64, 78),
    x_range=(43, 53),
) -> np.ndarray:
    y0, y1 = y_range
    x0, x1 = x_range
    
    top = obs[y0, x0:x1, :]
    bottom = obs[y1, x0:x1, :]
    left = obs[(y0+1):(y1-1), x0]
    right = obs[(y0+1):(y1-1), x1]
    return np.concatenate([top, bottom, left, right], axis=0)


def border_color_check(
    obs: np.ndarray,
    y_range: tuple[int, int],
    x_range: tuple[int, int],
    rgb_min_lim: np.ndarray,
    rgb_max_lim: np.ndarray
) -> bool:
    border = image_border(obs, y_range=y_range, x_range=x_range)
    within_min = np.all(border >= rgb_min_lim, axis=1)
    within_max = np.all(border <= rgb_max_lim, axis=1)
    return bool(np.all(within_min & within_max))
