import logging.config
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import yaml

from .handlers.tensorboard import BatchedTensorBoardHandler


def setup_logger(env_name: str, experiment_name: str, config_file_name="logging_config.yaml") -> None:
    config_file_path = Path(__file__).parent / config_file_name
    with open(config_file_path) as f:
        config = yaml.safe_load(f)

    log_dir = Path("logs") / env_name / experiment_name
    tb_log_dir = Path("logs") / env_name / "tensorboard" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(
        log_dir=tb_log_dir,
        flush_secs=300,     # TensorBoard disk flush
        max_queue=20000     # internal event buffer
    )

    tb_handler = BatchedTensorBoardHandler(
        writer,
        batch_size=10,
        flush_secs=5.0,
    )
    tb_handler.setLevel(logging.DEBUG)

    log_file = log_dir / "app.log"
    config["handlers"]["file"]["filename"] = str(log_file)
    logging.config.dictConfig(config)
    
    logging.getLogger().addHandler(tb_handler)
