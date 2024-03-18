"""Model Wrappers API."""

from eva.core.models.networks.wrappers.base import BaseModel
from eva.core.models.networks.wrappers.from_function import ModelFromFunction
from eva.core.models.networks.wrappers.huggingface import HuggingFaceModel
from eva.core.models.networks.wrappers.onnx import ONNXModel

__all__ = ["BaseModel", "ModelFromFunction", "HuggingFaceModel", "ONNXModel"]
