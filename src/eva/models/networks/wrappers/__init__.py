"""Model Wrappers API."""

from eva.models.networks.wrappers.base import BaseModel
from eva.models.networks.wrappers.from_function import ModelFromFunction
from eva.models.networks.wrappers.huggingface import HuggingFaceModel
from eva.models.networks.wrappers.onnx import ONNXModel

__all__ = ["BaseModel", "ModelFromFunction", "HuggingFaceModel", "ONNXModel"]
