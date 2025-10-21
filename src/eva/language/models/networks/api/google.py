"""Models from Google."""

import os

from eva.language.models import wrappers
from eva.language.models.networks.registry import model_registry


class _Gemini(wrappers.LiteLLMModel):
    """Base class for Gemini models."""

    def __init__(self, model_name: str, api_kwargs: dict | None = None):
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY env variable must be set.")

        super().__init__(model_name=model_name, model_kwargs=api_kwargs)


@model_registry.register("google/gemini-2.5-flash-lite")
class Gemini25FlashLite(_Gemini):
    """Gemini 2.5 Flash Lite model."""

    def __init__(self):
        """Initializes the model."""
        super().__init__(model_name="gemini/gemini-2.5-flash-lite")
