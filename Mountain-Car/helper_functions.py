import gymnasium as gym # type: ignore
from gymnasium.wrappers import AtariPreprocessing, RecordVideo, Autoreset # type: ignore
import os
import torch as T



def make_env(env_id: str, batch_size: int = 1, do_preprocessing: bool = False, make_video: bool = False, single_env: bool = False):
    '''
    env = make_env("CartPole-v1")
    '''
    if single_env:
        env = gym.make(env_id, render_mode="rgb_array")
    else:
        env = gym.vector.AsyncVectorEnv(
            [lambda: Autoreset(gym.make(env_id, render_mode="rgb_array")) for _ in range(batch_size)]
                                        )
    if do_preprocessing:
        env = AtariPreprocessing(env,
                                frame_skip=1,
                                screen_size=84,
                                terminal_on_life_loss=True,
                                grayscale_obs=False
                                )
    if make_video and batch_size == 1:
        os.makedirs('save_videos//ALE', exist_ok=True)
        env = RecordVideo(
            env,
            video_folder='.//save_videos',
            fps=20
                          )
        
    return env

def play_game(env_id, agent, distribution: str = "normal", temperature: float = 1, make_video: bool = True, repeat: int = 1, single_env: bool = False):
    env = make_env(env_id, make_video=make_video, single_env=single_env)
    step = 0
    total_reward = 0
    for _ in range(repeat):
        state, _ = env.reset()
        state = T.tensor(state, dtype=T.float32)
        state = state.unsqueeze(0)
        done = False
        while not done:
            with T.no_grad():
                action, _, _, _ = agent.action(state, distribution="normal", temperature=0.01)
            next_state, reward, terminated, truncated, info = env.step([float(action),])
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
        print(f"Steps {step}, result {result}, total reward {total_reward:.2f}, term {terminated}, trunc {truncated}, info {info}, final state {state[0].numpy()}, last action {float(action):.4f}")
    env.close()
    return total_reward, terminated
