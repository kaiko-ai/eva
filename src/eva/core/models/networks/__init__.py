"""Networks API."""

from eva.core.models.networks.linear import LinearClassifier
from eva.core.models.networks.mlp import MLP

__all__ = ["MLP", "LinearClassifier"]
