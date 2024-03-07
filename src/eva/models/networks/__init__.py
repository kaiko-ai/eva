"""Networks API."""

from eva.models.networks.mlp import MLP
from eva.models.wrappers.from_function import ModelFromFunction

__all__ = ["ModelFromFunction", "MLP"]
