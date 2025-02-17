from worker import Worker
import torch as T
import os

if __name__ == "__main__":
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    os.environ["MUJOCO_GL"] = "egl" if T.cuda.is_available() else 'osmesa'
    worker = Worker(
        env_id="Swimmer-v5",
        num_envs=32,
        hidden_dim=32,
        timesteps=32, 
        max_steps=int(1e8),
        learning_rate=0.0002,
        device=device
        )
    worker.train()
    