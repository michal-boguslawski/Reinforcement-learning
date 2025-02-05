import gymnasium as gym # type: ignore
from gymnasium.wrappers import AtariPreprocessing, RecordVideo # type: ignore
import os
import torch as T


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
            fps=20
                          )
        
    return env

def play_game(env_id, agent, distribution: str = "normal", temperature: float = 1, make_video: bool = True):
    env = make_env(env_id, make_video=make_video)
    state, _ = env.reset()
    state = T.tensor(state, dtype=T.float32)
    state = state.unsqueeze(0)
    done = False
    step = 0
    total_reward = 0
    while not done:
        action, _, _, _ = agent.action(state, distribution="normal", temperature=0.01)
        next_state, reward, terminated, truncated, info = env.step([float(action), ])
        next_state = T.tensor(next_state, dtype=T.float32)
        next_state = next_state.unsqueeze(0)
        done = terminated or truncated
        state = next_state
        step += 1
        total_reward += reward
    if terminated:
        result = 'Win'
    else:
        result = 'Lose'
    if make_video:
        print(f"Steps {step}, result {result}, total reward {total_reward:.2f}, term {terminated}, trunc {truncated}, info {info}, final state {state[0].numpy()}")
    env.close()
    return total_reward, terminated
