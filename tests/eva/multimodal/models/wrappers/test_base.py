"""Tests for VisionLanguageModel base class."""

from typing import Any, List
from unittest.mock import MagicMock

from typing_extensions import override

from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers.base import VisionLanguageModel


class ConcreteVisionLanguageModel(VisionLanguageModel):
    """Concrete implementation for testing."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize test model."""
        super().__init__(system_prompt=system_prompt)
        self.model = MagicMock()

    @override
    def format_inputs(self, batch: TextImageBatch) -> Any:
        return {"formatted": batch}

    @override
    def model_forward(self, batch: Any) -> List[str]:
        return ["response1", "response2"]


def test_system_prompt_initialization():
    """Test that system prompt is correctly initialized."""
    model_with_prompt = ConcreteVisionLanguageModel(system_prompt="You are a helpful assistant")
    assert model_with_prompt.system_message is not None
    assert model_with_prompt.system_message.content == "You are a helpful assistant"

    model_without_prompt = ConcreteVisionLanguageModel(system_prompt=None)
    assert model_without_prompt.system_message is None


def test_forward_delegates_to_format_and_model_forward():
    """Test that forward correctly delegates to format_inputs and model_forward."""
    model = ConcreteVisionLanguageModel()
    batch = MagicMock(spec=TextImageBatch)

    result = model.forward(batch)

    assert result == ["response1", "response2"]
