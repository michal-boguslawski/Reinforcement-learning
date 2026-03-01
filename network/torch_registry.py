from torch import nn


ACTIVATION_FUNCTIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}