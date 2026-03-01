import os
import shutil
import torch as T

from config.config import ExperimentConfig
from config.logging import setup_logger
from worker.worker import Worker


os.environ["MUJOCO_GL"] = "egl" if T.cuda.is_available() else "osmesa"


if __name__ == "__main__":
    # policy_name = "ppo"
    config_instance = ExperimentConfig()
    config = config_instance.get_config()
    env_name = config["env_name"]
    experiment_name = config["experiment_name"]

    logs_path = f"logs/{env_name}/{experiment_name}"
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)

    tensorboard_path = f"logs/{env_name}/tensorboard/{experiment_name}"
    if os.path.exists(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    os.makedirs(logs_path, exist_ok=True)

    setup_logger(env_name, experiment_name)
    
    config_instance.save_config(os.path.join(logs_path, "config.yaml"))

    worker = Worker(
        experiment_name=f"{env_name}/{experiment_name}",
        env_config=config["env_kwargs"],
        policy_config=config["policy"],
        network_config=config.get("network", {}),
        **config["worker_kwargs"]
    )
    worker.train(**config["train_kwargs"])
