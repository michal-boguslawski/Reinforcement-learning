from worker import WorkerNStep
import torch as T

if __name__ == "__main__":
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    worker = WorkerNStep(
        env_id="MountainCarContinuous-v0",
        timesteps=64, 
        device=device,
        episodes=10000
        )
    worker.train()
    