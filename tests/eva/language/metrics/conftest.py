"""Shared fixtures for language model tests."""

from typing import Any

import pytest
from typing_extensions import override

from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.models.wrappers.base import LanguageModel


class DummyJudgeModel(LanguageModel):
    """Dummy language model that returns configurable responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize the dummy model.

        Args:
            responses: List of response strings to return. If None, returns
                generic JSON responses with scores.
        """
        super().__init__(system_prompt=None)
        self.model = self.load_model()
        self.responses = responses or [
            '{"score": 8, "reason": "The response is accurate and well-structured."}',
            '{"score": 5, "reason": "The response is partially correct but lacks detail."}',
        ]

    @override
    def load_model(self) -> Any:
        return lambda x: x

    @override
    def format_inputs(self, batch: TextBatch) -> TextBatch:
        return batch

    @override
    def model_forward(self, batch: TextBatch) -> ModelOutput:
        generated_texts = [self.responses[i % len(self.responses)] for i in range(len(batch.text))]
        return {"generated_text": generated_texts}


@pytest.fixture
def dummy_judge_model() -> DummyJudgeModel:
    """Fixture providing a dummy judge model instance."""
    return DummyJudgeModel()
