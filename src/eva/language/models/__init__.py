"""Language Models API."""

from eva.language.models import modules, networks, wrappers
from eva.language.models.modules import LanguageModule, OfflineLanguageModule
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
        "OfflineLanguageModule",
    ]
except ImportError:
    __all__ = [
        "modules",
        "wrappers",
        "networks",
        "HuggingFaceModel",
        "LiteLLMModel",
        "LanguageModule",
        "OfflineLanguageModule",
    ]
