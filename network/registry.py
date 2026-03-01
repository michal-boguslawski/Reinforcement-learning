# Distributions
from .distributions.categorical import CategoricalDistribution
from .distributions.normal import NormalDistribution, MultivariateNormalDistribution

# Transforms
from .distributions.transforms import TanhAffineTransform

# Heads
from .heads.actor_critic import ActorCriticHead
from .heads.actor import ActorHead

# Backbones
from .backbones.mlp import MLPNetwork
from .backbones.cnn import SimpleCNN, CNN

# Cores
from .cores.identity import IdentityCore
from .cores.lstm import LSTMCore
from .cores.gru import GRUCore


DISTRIBUTIONS = {
    "normal": NormalDistribution,
    "categorical": CategoricalDistribution,
    "mvn": MultivariateNormalDistribution,
}


TRANSFORMS = {
    "normal": TanhAffineTransform,
    "mvn": TanhAffineTransform,
}


HEADS = {
    "actor_critic": ActorCriticHead,
    "actor": ActorHead,
}


BACKBONES = {
    "mlp": MLPNetwork,
    "simple_cnn": SimpleCNN,
    "cnn": CNN,
}


CORES = {
    "identity": IdentityCore,
    "lstm": LSTMCore,
    "gru": GRUCore,
}
