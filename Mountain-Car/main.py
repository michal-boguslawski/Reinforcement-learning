from worker import WorkerNStep
import torch as T

if __name__ == "__main__":
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    worker = WorkerNStep(
        env_id="MountainCarContinuous-v0", 
        batch_size=128,
        timesteps=20, 
        device=device
        )
    worker.train(episodes=800)