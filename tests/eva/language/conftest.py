"""Shared fixtures for language model tests."""

from typing import Any

import pytest
from typing_extensions import override

from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.models.wrappers.base import LanguageModel


class DummyLanguageModel(LanguageModel):
    """Dummy language model that returns configurable responses.

    This model is useful for testing without requiring actual API calls
    or model inference.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize the dummy model.

        Args:
            responses: List of response strings to return. If None, returns
                generic JSON responses with scores.
        """
        super().__init__(system_prompt=None)
        self.model = lambda x: x  # Dummy model callable
        self.responses = responses or [
            '{"score": 8, "reason": "The response is accurate and well-structured."}',
            '{"score": 5, "reason": "The response is partially correct but lacks detail."}',
        ]

    @override
    def load_model(self) -> Any:
        """Load the dummy model."""
        return lambda x: x

    @override
    def format_inputs(self, batch: TextBatch) -> TextBatch:
        """Pass through inputs without modification."""
        return batch

    @override
    def model_forward(self, batch: TextBatch) -> ModelOutput:
        """Generate responses for the given batch.

        Cycles through the configured responses for each item in the batch.
        """
        generated_texts = [self.responses[i % len(self.responses)] for i in range(len(batch.text))]
        return {"generated_text": generated_texts}


@pytest.fixture
def dummy_language_model() -> DummyLanguageModel:
    """Provide a dummy language model for testing.

    Returns:
        A DummyLanguageModel instance with default JSON responses.
    """
    return DummyLanguageModel()
