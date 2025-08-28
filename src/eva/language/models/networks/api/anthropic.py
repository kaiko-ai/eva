"""Models from Anthropic."""

import os

from eva.language.models import wrappers
from eva.language.models.networks.registry import model_registry


class _Claude(wrappers.LiteLLMModel):
    """Base class for Claude models."""

    def __init__(self, model_name: str, system_prompt: str | None = None):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY env variable must be set.")

        super().__init__(model_name=model_name, system_prompt=system_prompt)


@model_registry.register("anthropic/claude-3-5-sonnet-20240620")
class Claude35Sonnet20240620(_Claude):
    """Claude 3.5 Sonnet (June 2024) model."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize the model."""
        super().__init__(model_name="claude-3-5-sonnet-20240620", system_prompt=system_prompt)


@model_registry.register("anthropic/claude-3-7-sonnet-20250219")
class Claude37Sonnet20250219(_Claude):
    """Claude 3.7 Sonnet (February 2025) model."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize the model."""
        super().__init__(model_name="claude-3-7-sonnet-20250219", system_prompt=system_prompt)
