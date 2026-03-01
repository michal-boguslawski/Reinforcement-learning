# Exploration methods
from .egreedy import EGreedyExploration
from .distribution import DistributionExploration


EXPLORATIONS = {
    "egreedy": EGreedyExploration,
    "distribution": DistributionExploration,
}
