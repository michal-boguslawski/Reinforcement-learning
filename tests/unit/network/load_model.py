import os
from pathlib import Path
import torch as T

from network.model import RLModel


def load_model() -> RLModel:
    folder_path = Path(__file__).parent.absolute()
    backbone_state_dict = T.load(os.path.join(folder_path, "data", "backbone_state_dict.pth"))
    head_state_dict = T.load(os.path.join(folder_path, "data", "head_state_dict.pth"))


    model = RLModel(
        input_shape=(96, 96, 3),
        num_actions=3,
        num_features=256,
        backbone_name="simple_cnn",
        distribution="mvn",
        low=T.tensor([-1., 0., 0.]),
        high=T.tensor([1., 1., 1.]),
    )
    model.backbone.load_state_dict(backbone_state_dict)
    model.head.load_state_dict(head_state_dict)
    return model
