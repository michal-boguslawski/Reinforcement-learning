from typing import Dict, Any
from .registry import EXPLORATIONS
from .base import BaseExploration


def get_exploration(
    exploration_method_name: str,
    exploration_kwargs: Dict[str, Any]
) -> BaseExploration:
    return EXPLORATIONS[exploration_method_name](**exploration_kwargs)
