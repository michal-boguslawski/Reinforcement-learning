from worker import WorkerNStep
import torch as T
import multiprocessing

if __name__ == "__main__":
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    multiprocessing.set_start_method("spawn", force=True)
    worker = WorkerNStep(
        env_id="MountainCarContinuous-v0",
        batch_size=16,
        timesteps=32, 
        device=device
        )
    worker.train()
    