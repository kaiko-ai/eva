"""Language Models API."""

from eva.language.models import modules, networks, wrappers
from eva.language.models.modules import LanguageModule
from eva.language.models.wrappers import HuggingFaceTextModel, LiteLLMModel

try:
    from eva.language.models.wrappers import VLLMTextModel

    __all__ = [
        "modules",
        "wrappers",
        "networks",
        "HuggingFaceTextModel",
        "LiteLLMModel",
        "VLLMTextModel",
        "LanguageModule",
    ]
except ImportError:
    __all__ = [
        "modules",
        "wrappers",
        "networks",
        "HuggingFaceTextModel",
        "LiteLLMModel",
        "LanguageModule",
    ]
