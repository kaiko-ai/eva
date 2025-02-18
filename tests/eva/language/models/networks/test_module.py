"""Tests the TextModule module."""

import pytest
from torch import nn

from eva.language.models import TextModule


def test_forward(text_module, text_model):
    """Test the forward method of the TextModule class."""
    input_text = "Hello world"
    expected = text_model.generate(input_text)
    result = text_module.forward(input_text)
    assert result == expected


def test_validation_step(text_module, text_model):
    """Test the validation_step method of the TextModule class."""
    data = "What is the capital of France?"
    targets = "Paris"
    metadata = {"id": 1}
    batch = (data, targets, metadata)

    expected_message = text_module.prompt + str(data) + "\nAnswer: "
    expected_predictions = text_model.generate(expected_message)

    output = text_module.validation_step(batch)

    assert "predictions" in output
    assert "targets" in output
    assert "metadata" in output
    assert output["predictions"] == expected_predictions
    assert output["targets"] == targets
    assert output["metadata"] == metadata


def test_init_attributes(text_model):
    """Test the attributes of the TextModule class."""
    prompt = "Initialization Prompt: "
    module_instance = TextModule(model=text_model, prompt=prompt)
    assert module_instance.model is text_model
    assert module_instance.prompt == prompt


class TextModel(nn.Module):
    """A simple text model for testing purposes."""

    def generate(self, prompts: str):
        """Generate some text based on the input prompt."""
        return [f"Generated: {prompts}"]


@pytest.fixture
def text_model():
    """Return a TextModel instance."""
    return TextModel()


@pytest.fixture
def text_module(text_model):
    """Return a TextModule instance."""
    prompt = "Test Prompt: "
    return TextModule(model=text_model, prompt=prompt)
