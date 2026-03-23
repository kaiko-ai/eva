"""Model Wrappers API."""

from eva.core.models.wrappers.base import BaseModel
from eva.core.models.wrappers.from_function import ModelFromFunction
from eva.core.models.wrappers.from_torchhub import TorchHubModel
from eva.core.models.wrappers.huggingface import HuggingFaceModel
from eva.core.models.wrappers.onnx import ONNXModel

__all__ = [
    "BaseModel",
    "HuggingFaceModel",
    "ModelFromFunction",
    "ONNXModel",
    "TorchHubModel",
]
