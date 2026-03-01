import logging
import numpy as np
import os
import torch as T
from torch import nn

from .factories import make_action_distribution, make_backbone, make_head, make_core
from .models.models import ModelOutput


logger = logging.getLogger(__name__)


class RLModel(nn.Module):
    def __init__(
        self,
        input_shape: tuple,                     # automatically derived
        num_actions: int,                       # automatically derived
        num_features: int = 64,

        backbone_name: str = "mlp",
        backbone_kwargs: dict = {},

        core_name: str = "identity",
        core_kwargs: dict = {},

        head_name: str = "actor_critic",
        head_kwargs: dict = {},
        
        distribution: str = "categorical",
        low: T.Tensor | None = None,            # automatically derived
        high: T.Tensor | None = None,           # automatically derived
        initial_deviation: float = 1.0,
        device: T.device = T.device("cpu"),

        weights_kwargs: dict | None = None,

        *args,
        **kwargs
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_features = num_features

        self.backbone_name = backbone_name
        self.backbone_kwargs = backbone_kwargs

        self.core_name = core_name
        self.core_kwargs = core_kwargs

        self.head_name = head_name
        self.head_kwargs = head_kwargs
        
        self.distribution = distribution
        self.initial_deviation = initial_deviation
        self.high = T.as_tensor(high, device=device) if high is not None else high
        self.low = T.as_tensor(low, device=device) if low is not None else low

        self._setup()

        self.device = device
        self.to(device)
        if weights_kwargs:
            self.load_weights(**weights_kwargs)

    def _setup(self):
        self._setup_dist()
        self._setup_backbone()
        self._setup_core()
        self._setup_head()

    def _setup_backbone(self):
        self.backbone = make_backbone(
            backbone_name=self.backbone_name,
            input_shape=self.input_shape,
            **self.backbone_kwargs
        )

    def _setup_core(self):
        self.core = make_core(
            core_name=self.core_name,
            in_features=self.backbone.out_features,
            **self.core_kwargs
        )

    def _setup_head(self):
        self.head = make_head(
            head_name=self.head_name,
            num_actions=self.num_actions,
            in_features=self.core.out_features,
            **self.head_kwargs
        )

    def _setup_dist(self):
        self.log_std = nn.Parameter(T.ones((self.num_actions, )) * np.log(self.initial_deviation))
        self.raw_scale_tril = nn.Parameter(T.eye(self.num_actions) * (self.initial_deviation ** (1/2)))
        self.dist = make_action_distribution(
            dist_name=self.distribution,
            log_std=self.log_std,
            raw_scale_tril=self.raw_scale_tril,
            high=self.high,
            low=self.low,
        )

    def save_weights(self, folder_path: str):
        file_path = os.path.join(folder_path, "model.pt")
        T.save(self.state_dict(), file_path)
        logger.info(f"Saved model to {file_path}")

    def load_weights(
        self,
        file_path: str,
        param_groups: list[str] | None = None,
        strict: bool = True
    ):
        """
        Load full or partial model weights from a single state_dict file.

        Args:
            file_path: path to the saved full model .pt file.
            param_groups: list of submodules to load. Options: "backbone", "core", "head", "dist", "full".
                        If None, defaults to ["full"].
            strict: whether to enforce strict key matching when loading submodules.
            map_location: device mapping, e.g., 'cpu' or 'cuda'.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        if param_groups is None:
            param_groups = ["full"]

        state_dict = T.load(file_path, map_location=self.device)

        for group in param_groups:
            if group == "full":
                self.load_state_dict(state_dict, strict=strict)

            elif group in ["backbone", "core", "head"]:
                # Filter keys belonging to the module
                filtered_dict = {k.replace(f"{group}.", ""): v
                                for k, v in state_dict.items()
                                if k.startswith(f"{group}.")}
                submodule = getattr(self, group)
                submodule.load_state_dict(filtered_dict, strict=strict)

            elif group == "dist":
                # Distribution parameters are nn.Parameters at the top level
                with T.no_grad():
                    if "log_std" in state_dict:
                        self.log_std.copy_(state_dict["log_std"])
                    if "raw_scale_tril" in state_dict:
                        self.raw_scale_tril.copy_(state_dict["raw_scale_tril"])
            else:
                raise ValueError(f"Unknown param group '{group}'")

        logger.info(f"Loaded param_groups={param_groups} from {file_path}")

    def forward(self, input_tensor: T.Tensor, core_state: T.Tensor | None = None, temperature: float = 1.) -> ModelOutput:
        features = self.backbone(input_tensor=input_tensor)
        core = self.core(features=features.features, core_state=core_state)
        head_output = self.head(features=core.core_out)
        dist = self.dist(logits=head_output.actor_logits, temperature=temperature)
        return ModelOutput(
            actor_logits=head_output.actor_logits,
            critic_value=head_output.critic_value,
            dist=dist,
            core_state=core.core_state
        )
