from worker import Worker, WorkerNStep
import torch as T # type: ignore

if __name__ == "__main__":
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    worker = WorkerNStep(
        env_id="CartPole-v1", 
        batch_size=128,
        timesteps=20, 
        device=device
        )
    worker.train(episodes=1000)