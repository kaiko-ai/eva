"""Models from Anthropic."""

import os

from eva.multimodal.models import wrappers
from eva.multimodal.models.networks.registry import model_registry


class _Claude(wrappers.LiteLLMModel):
    """Base class for Claude models."""

    def __init__(self, model_name: str):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY env variable must be set.")

        super().__init__(model_name=model_name)


@model_registry.register("anthropic/claude-3-5-sonnet-20240620")
class Claude35Sonnet20240620(_Claude):
    def __init__(self):
        super().__init__(model_name="claude-3-5-sonnet-20240620")


@model_registry.register("anthropic/claude-3-7-sonnet-20250219")
class Claude37Sonnet20250219(_Claude):
    def __init__(self):
        super().__init__(model_name="claude-3-7-sonnet-20250219")
