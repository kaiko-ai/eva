"""Networks API."""

from eva.core.models.networks.mlp import MLP
from eva.core.models.networks.wrappers import HuggingFaceModel, ModelFromFunction, ONNXModel

__all__ = ["MLP", "ModelFromFunction", "HuggingFaceModel", "ONNXModel"]
