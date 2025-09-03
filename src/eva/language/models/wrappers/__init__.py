"""Language Model Wrappers API."""

from eva.language.models.wrappers.base import LanguageModel
from eva.language.models.wrappers.from_registry import ModelFromRegistry
from eva.language.models.wrappers.huggingface import HuggingFaceModel
from eva.language.models.wrappers.litellm import LiteLLMModel

try:
    from eva.language.models.wrappers.vllm import VllmModel

    __all__ = [
        "LanguageModel",
        "HuggingFaceModel",
        "LiteLLMModel",
        "VllmModel",
        "ModelFromRegistry",
    ]
except ImportError:
    __all__ = ["LanguageModel", "HuggingFaceModel", "LiteLLMModel", "ModelFromRegistry"]
