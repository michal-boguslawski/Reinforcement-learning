import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo, DtypeObservation, RecordEpisodeStatistics
import gymnasium.wrappers.vector as vec_wrappers
import numpy as np

from .env_utils import prepare_wrappers
from .registry import WRAPPERS


def make_vec(
    id: str,
    num_envs: int,
    training: bool = True,
    record: bool = False,
    video_folder: str | None = None,
    name_prefix: str | None = None,
    training_wrappers: dict | None = None,
    general_wrappers: dict | None = None,
    normalize_rewards: bool = False,
    normalize_gamma: float = 0.99,
    permute_observations: bool = False,
    verbose: int = 0,
    *args,
    **kwargs
) -> gym.vector.VectorEnv:
    wrappers = []
    wrappers.append(lambda env: DtypeObservation(env, np.float32))

    if ( not training ) or ( verbose == 1 ):
        wrappers.append(lambda env: RecordEpisodeStatistics(env))

    if record and video_folder is not None and name_prefix is not None:
        record_wrapper = lambda env: RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda x: True
        )
        kwargs["render_mode"] = "rgb_array"
        wrappers.append(record_wrapper)

    if training:
        wrappers.extend(prepare_wrappers(training_wrappers))

    wrappers.extend(prepare_wrappers(general_wrappers))

    if permute_observations:
        wrappers.append(WRAPPERS["permute_observations"])

    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    
    try:
        envs = gym.make_vec(
            id=id,
            num_envs=num_envs,
            vector_kwargs=None,
            wrappers=wrappers,
            *args,
            **kwargs
        )

        if normalize_rewards and training:
            envs = vec_wrappers.NormalizeReward(envs, gamma=normalize_gamma)

        envs = vec_wrappers.NumpyToTorch(envs)

        # if len(envs.observation_space.shape) == 4:
        #     envs = VecTransposeObservationWrapper(envs)
    except Exception as e:
        raise RuntimeError(f"Failed to create vectorized environment '{id}' with {num_envs} envs: {e}")
    
    return envs
