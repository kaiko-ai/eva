"""Language Models API."""

from eva.language.models import networks, wrappers
from eva.language.models.networks import TextModule
from eva.language.models.wrappers import HuggingFaceTextModel, LiteLLMTextModel

try:
    from eva.language.models.wrappers import VLLMTextModel
    __all__ = [
        "networks",
        "wrappers",
        "TextModule",
        "HuggingFaceTextModel",
        "LiteLLMTextModel",
        "VLLMTextModel",
    ]
except ImportError:
    __all__ = [
        "networks",
        "wrappers",
        "TextModule",
        "HuggingFaceTextModel",
        "LiteLLMTextModel",
    ]
