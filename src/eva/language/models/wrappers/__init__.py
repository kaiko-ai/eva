"""Language Model Wrappers API."""

from eva.language.models.wrappers.huggingface import HuggingFaceTextModel
from eva.language.models.wrappers.litellm import LiteLLMModel

try:
    from eva.language.models.wrappers.vllm import VLLMTextModel

    __all__ = ["HuggingFaceTextModel", "LiteLLMModel", "VLLMTextModel"]
except ImportError:
    __all__ = ["HuggingFaceTextModel", "LiteLLMModel"]
