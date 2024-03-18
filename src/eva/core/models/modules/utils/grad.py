"""Gradient related utilities and helper functions."""

from torch import nn


def deactivate_requires_grad(module: nn.Module) -> None:
    """Deactivates the `requires_grad` flag for all parameters of a model.

    Args:
        module: The torch module to deactivate the gradient computation in place.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def activate_requires_grad(module: nn.Module) -> None:
    """Activates the `requires_grad` flag for all parameters of a model.

    Args:
        module: The torch module to deactivate the gradient computation in place.
    """
    for parameter in module.parameters():
        parameter.requires_grad = True
