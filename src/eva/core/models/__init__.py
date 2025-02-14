"""Models API."""

from eva.core.models.modules import HeadModule, InferenceModule
from eva.core.models.networks import MLP, LinearClassifier
from eva.core.models.wrappers import (
    BaseModel,
    HuggingFaceModel,
    ModelFromFunction,
    ONNXModel,
    TorchHubModel,
)

__all__ = [
    "HeadModule",
    "InferenceModule",
    "MLP",
    "LinearClassifier",
    "BaseModel",
    "HuggingFaceModel",
    "ModelFromFunction",
    "ONNXModel",
    "TorchHubModel",
]
