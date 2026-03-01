import ast
import gymnasium as gym
import logging
import numpy as np
import torch as T

from .utils import border_color_check


logger = logging.getLogger(__name__)


class TerminalBonusWrapper(gym.Wrapper):
    def __init__(self, env, terminated_bonus: float | None = 0., truncated_bonus: float | None = 0.):
        super().__init__(env)
        self.terminated_bonus = terminated_bonus or 0
        self.truncated_bonus = truncated_bonus or 0
        logger.info(f"TerminalBonusWrapper attached with params {terminated_bonus} {truncated_bonus}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add extra reward when the episode terminates
        reward = float(reward)
        if terminated:
            reward += self.terminated_bonus
        if truncated:
            reward += self.truncated_bonus

        return obs, reward, terminated, truncated, info


class PowerObsRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        pow_factors: T.Tensor | None = None,
        abs_factors: T.Tensor | None = None,
        nominal_factors: T.Tensor | None = None,
        decay_factor: float | None = 1
    ):
        super().__init__(env)
        self.pow_factors = pow_factors
        self.abs_factors = abs_factors
        self.nominal_factors = nominal_factors
        self.decay = 1
        self.decay_factor = decay_factor or 1
        logger.info(f"PowerObsRewardWrapper attached with params {pow_factors} {abs_factors} {nominal_factors} {decay_factor}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.pow_factors is not None:
            reward += (obs ** 2 * self.pow_factors).sum().item() * self.decay
        if self.abs_factors is not None:
            reward += (np.abs(obs) * self.abs_factors).sum().item() * self.decay
        if self.nominal_factors is not None:
            reward += (obs * self.nominal_factors).sum().item() * self.decay
        if terminated:
            self.decay *= self.decay_factor
        
        return obs, reward, terminated, truncated, info


class NoMovementInvPunishmentRewardWrapper(gym.Wrapper):
    def __init__(self, env, punishment: T.Tensor):
        super().__init__(env)
        self.punishment = punishment

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        inv_obs = 1 / (np.abs(obs) + 1e-6)
        reward -= (inv_obs.clip(0, 100) * self.punishment).sum().item()

        return obs, reward, terminated, truncated, info


class NoMovementTruncateWrapper(gym.Wrapper):
    def __init__(self, env, index: int, penalty: float | int = 10., steps: int = 50, eps: float = 1e-3):
        super().__init__(env)
        self.index = index
        self.steps = steps
        self.eps = eps
        self.penalty = penalty
        self._counter = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        index_observation = obs[self.index]
        if abs(index_observation) < self.eps:
            self._counter += 1
        else:
            self._counter = 0
        if self._counter > self.steps:
            truncated = True
            reward = float(reward) - self.penalty
            self._counter = 0
            logger.info("Env truncated due to lack of movement")

        return obs, reward, terminated, truncated, info


class VecTransposeObservationWrapper(gym.vector.VectorObservationWrapper):
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)

        b, h, w, c = obs_space.shape
        self.observation_space = gym.spaces.Box(
            low=obs_space.low.transpose(0, 3, 1, 2),
            high=obs_space.high.transpose(0, 3, 1, 2),
            shape=(b, c, h, w),
            dtype=obs_space.dtype, # type: ignore
        )

    def observations(self, observations):
        return observations.permute(0, 3, 1, 2).contiguous()


class TransposeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)

        h, w, c = obs_space.shape
        self.observation_space = gym.spaces.Box(
            low=obs_space.low.transpose(2, 0, 1),
            high=obs_space.high.transpose(2, 0, 1),
            shape=(c, h, w),
            dtype=obs_space.dtype, # type: ignore
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class ActionPowerRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        pow_factors: T.Tensor | None = None,
        abs_factors: T.Tensor | None = None,
        decay_factor: float | None = 1
    ):
        super().__init__(env)
        self.pow_factors = pow_factors
        self.abs_factors = abs_factors
        self.decay = 1
        self.decay_factor = decay_factor or 1
        logger.info(f"ActionPowerRewardWrapper attached with params {pow_factors} {abs_factors} {decay_factor}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.pow_factors is not None:
            reward += (action ** 2 * self.pow_factors).sum().item() * self.decay
        if self.abs_factors is not None:
            reward += (np.abs(action) * self.abs_factors).sum().item() * self.decay
        if terminated:
            self.decay *= self.decay_factor
        
        return obs, reward, terminated, truncated, info


class ActionInteractionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        factors: dict,
    ):
        super().__init__(env)
        self.factors = self._parse_factors(factors)
        logger.info(f"ActionInteractionWrapper attached with params {self.factors}")

    def _parse_factors(self, factors: dict) -> np.ndarray:
        env_shape = self.env.action_space.shape
        if env_shape is None:
            raise ValueError("No valid shape for environment")
        n = env_shape[-1]
        squared_tensor = np.zeros((n, n))
        for key, value in factors.items():
            if isinstance(key, str):
                key = ast.literal_eval(key)
            squared_tensor[key[0], key[1]] = value
        return squared_tensor

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.factors is not None:
            abs_action = np.abs(action)
            reward += float((abs_action @ self.factors.T) @ abs_action.T)
        
        return obs, reward, terminated, truncated, info


class OutOfTrackPenaltyAndTerminationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        y_range=(63, 79),
        x_range=(42, 54),
        rgb_max_lim=np.array([120, 255, 120]),
        rgb_min_lim=np.array([0, 180, 0]),
        out_of_track_penalty=1,
        termination_penalty=100,
        terminate_after=10,
        start_at_step=20,
        *args,
        **kwargs
    ):
        super().__init__(env)
        self.y_range = y_range
        self.x_range = x_range
        self.rgb_max_lim = rgb_max_lim
        self.rgb_min_lim = rgb_min_lim
        self.out_of_track_penalty = out_of_track_penalty
        self.termination_penalty = termination_penalty
        self.terminate_after = terminate_after
        self.start_at_step = start_at_step
        self.step_counter = 0
        self.counter = 0
        self.current_penalty = 0
        logger.info(f"OutOfTrackPenaltyAndTerminationWrapper attached with penalty {self.out_of_track_penalty}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_counter = 0
        self.counter = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        if self.step_counter < self.start_at_step:
            return obs, reward, terminated, truncated, info

        out_of_track = border_color_check(
            obs, self.y_range, self.x_range, self.rgb_min_lim, self.rgb_max_lim
        )
        
        if out_of_track:
            self.counter += 1
            reward = float(reward)
            reward -= self.out_of_track_penalty

            terminated = terminated or (self.counter > self.terminate_after)
            if terminated:
                reward -= self.termination_penalty
        else:
            self.counter = 0
        
        return obs, reward, terminated, truncated, info


class ObservationsInteractionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        factors: dict,
    ):
        super().__init__(env)
        self.factors = self._parse_factors(factors)
        logger.info(f"PowerObsRewardWrapper attached with params {self.factors}")

    def _parse_factors(self, factors: dict) -> np.ndarray:
        env_shape = self.env.observation_space.shape
        if env_shape is None:
            raise ValueError("No valid shape for environment")
        n = env_shape[-1]
        squared_tensor = np.zeros((n, n))
        for key, value in factors.items():
            if isinstance(key, str):
                key = ast.literal_eval(key)
            squared_tensor[key[0], key[1]] = value
        return squared_tensor

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.factors is not None:
            reward += float((obs @ self.factors.T) @ obs.T)
        
        return obs, reward, terminated, truncated, info
