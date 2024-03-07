"""Model Wrappers API."""

from eva.models.wrappers.base import BaseModel
from eva.models.wrappers.from_function import ModelFromFunction
from eva.models.wrappers.huggingface import HuggingFaceModel
from eva.models.wrappers.onnx import ONNXModel

__all__ = ["BaseModel", "HuggingFaceModel", "ONNXModel", "ModelFromFunction"]
