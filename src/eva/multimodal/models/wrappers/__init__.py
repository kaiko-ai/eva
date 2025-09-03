"""Multimodal Wrapper API."""

from eva.multimodal.models.wrappers.base import VisionLanguageModel
from eva.multimodal.models.wrappers.from_registry import ModelFromRegistry
from eva.multimodal.models.wrappers.huggingface import HuggingFaceModel
from eva.multimodal.models.wrappers.litellm import LiteLLMModel

__all__ = [
    "HuggingFaceModel",
    "LiteLLMModel",
    "ModelFromRegistry",
    "VisionLanguageModel",
]
