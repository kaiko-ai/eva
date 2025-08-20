"""Language Models API."""

from eva.language.models import modules, networks, wrappers
from eva.language.models.modules import LanguageModule
from eva.language.models.wrappers import HuggingFaceModel, LiteLLMModel

try:
    from eva.language.models.wrappers import VllmModel

    __all__ = [
        "modules",
        "wrappers",
        "networks",
        "HuggingFaceModel",
        "LiteLLMModel",
        "VllmModel",
        "LanguageModule",
    ]
except ImportError:
    __all__ = [
        "modules",
        "wrappers",
        "networks",
        "HuggingFaceModel",
        "LiteLLMModel",
        "LanguageModule",
    ]
