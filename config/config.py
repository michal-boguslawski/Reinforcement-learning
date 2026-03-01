import os
from pathlib import Path
import torch as T
from typing import Any
import yaml

from .models import ExperimentConfigModel


class ExperimentConfig:
    def __init__(self, config_path: str = os.path.join(Path(__file__).parent.absolute(), "config.yaml")):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
        self.config = ExperimentConfigModel.model_validate(data)

    def get_config(self) -> dict[str, Any]:
        return self.config.model_dump(exclude_none=True)

    def save_config(self, write_path: str):
        with open(write_path, "w") as f:
            yaml.dump(self.config.model_dump(exclude_none=True), f)
