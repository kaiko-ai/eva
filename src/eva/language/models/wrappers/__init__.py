"""Language Model Wrappers API."""

from eva.language.models.wrappers.base import LanguageModel
from eva.language.models.wrappers.from_registry import ModelFromRegistry
from eva.language.models.wrappers.huggingface import HuggingFaceModel
from eva.language.models.wrappers.litellm import LiteLLMModel

__all__ = [
    "LanguageModel",
    "HuggingFaceModel",
    "LiteLLMModel",
    "ModelFromRegistry",
]
try:
    from eva.language.models.wrappers.vllm import VllmModel  # noqa: F401

    __all__.append("VllmModel")
except ImportError:
    pass
