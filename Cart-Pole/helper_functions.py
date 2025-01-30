import gymnasium as gym # type: ignore
from gymnasium.wrappers import AtariPreprocessing, RecordVideo # type: ignore
import os
import torch as T # type: ignore


def make_env(env_id: str, do_preprocessing: bool = False, make_video: bool = False):
    '''
    env = make_env("CartPole-v1")
    '''
    env = gym.make(env_id, render_mode="rgb_array")
    if do_preprocessing:
        env = AtariPreprocessing(env,
                                frame_skip=1,
                                screen_size=84,
                                terminal_on_life_loss=True,
                                grayscale_obs=False
                                )
    if make_video:
        os.makedirs('save_videos//ALE', exist_ok=True)
        env = RecordVideo(
            env,
            video_folder='.//save_videos',
            fps=10
                          )
        
    return env

def play_game(env_id, agent, strategy: str = 'argmax', temperature: float = 1):
    env = make_env(env_id, make_video=True)
    state, _ = env.reset()
    state = T.tensor(state, dtype=T.float32)
    state = state.unsqueeze(0)
    done = False
    step = 0
    while not done:
        action = agent.action(state, strategy=strategy, temperature=temperature)
        next_state, _, terminated, truncated, _ = env.step(int(action))
        next_state = T.tensor(next_state, dtype=T.float32)
        next_state = next_state.unsqueeze(0)
        done = terminated or truncated
        state = next_state
        step += 1
    if terminated:
        result = 'Lose'
    else:
        result = 'Win'
    
    print(f"Steps {step}, result {result}")
    env.close()
