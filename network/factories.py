from torch import nn

from .registry import DISTRIBUTIONS, TRANSFORMS, BACKBONES, HEADS, CORES
from .distributions.base import ActionDistribution


def make_action_distribution(
    dist_name: str,
    *args,
    **kwargs,
) -> ActionDistribution:
    transform = None

    try:
        base_cls = DISTRIBUTIONS[dist_name]
    except KeyError:
        raise ValueError(f"Unknown distribution: {dist_name}")

    base = base_cls(**kwargs)

    transform_cls = TRANSFORMS.get(dist_name, None)
    if transform_cls:
        transform = transform_cls(**kwargs)

    return ActionDistribution(base, transform)


def make_backbone(
    backbone_name: str,
    input_shape: tuple,
    **kwargs
) -> nn.Module:
    backbone = BACKBONES[backbone_name](
        input_shape=input_shape,
        **kwargs
    )
    return backbone


def make_head(
    head_name: str,
    num_actions: int,
    in_features: int,
    *args,
    **kwargs
) -> nn.Module:
    head = HEADS[head_name](
        num_actions=num_actions,
        in_features=in_features,
        **kwargs
    )
    return head


def make_core(
    core_name: str,
    in_features: int,
    *args,
    **kwargs
) -> nn.Module:
    core = CORES[core_name](
        in_features=in_features,
        **kwargs
    )
    return core
