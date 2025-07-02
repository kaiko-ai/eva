"""Tests the TextModule module."""

import pytest
from torch import nn

from eva.language.models import TextModule


def test_forward(text_module, text_model):
    """Test the forward method of the TextModule class."""
    input_text = "Hello world"
    expected = text_model(input_text)
    result = text_module.forward(input_text)
    assert result == expected


def test_validation_step(text_module, text_model):
    """Test the validation_step method of the TextModule class."""
    data = ["What is the capital of France?"]
    targets = ["Paris"]
    metadata = [{"id": 1}]
    batch = (data, targets, metadata)

    # The module creates messages list: [str(d) + "\n" + prompt for d in data]
    expected_messages = [str(data[0]) + "\n" + text_module.prompt]
    expected_predictions = text_model(expected_messages)

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

    def forward(self, prompts):
        """Generate some text based on the input prompt."""
        if isinstance(prompts, str):
            return [f"Generated: {prompts}"]
        elif isinstance(prompts, list):
            return [f"Generated: {prompt}" for prompt in prompts]
        else:
            return [f"Generated: {str(prompts)}"]


@pytest.fixture
def text_model():
    """Return a TextModel instance."""
    return TextModel()


@pytest.fixture
def text_module(text_model):
    """Return a TextModule instance."""
    prompt = "Test Prompt: "
    return TextModule(model=text_model, prompt=prompt)
