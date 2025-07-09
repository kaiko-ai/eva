"""Language Models API."""

from eva.language.models import modules, wrappers
from eva.language.models.modules import TextModule
from eva.language.models.wrappers import HuggingFaceTextModel, LiteLLMTextModel

try:
    from eva.language.models.wrappers import VLLMTextModel

    __all__ = [
        "modules",
        "wrappers",
        "TextModule",
        "HuggingFaceTextModel",
        "LiteLLMTextModel",
        "VLLMTextModel",
    ]
except ImportError:
    __all__ = [
        "modules",
        "wrappers",
        "TextModule",
        "HuggingFaceTextModel",
        "LiteLLMTextModel",
    ]
