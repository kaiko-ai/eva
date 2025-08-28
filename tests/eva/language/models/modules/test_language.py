"""Tests the language model module."""

import pytest
from torch import nn

from eva.language.models import LanguageModule
from eva.language.models.typings import TextBatch


def test_forward(language_module):
    """Test the forward method of the LanguageModule class."""
    input_text = ["Hello world"]
    expected = ["Dummy response Nr. 0"]
    result = language_module.forward(input_text)
    assert result == expected


def test_validation_step(language_module, model):
    """Test the validation_step method of the LanguageModule class."""
    data = ["What is the capital of France?"]
    targets = ["Paris"]
    metadata = [{"id": 1}]
    batch = (data, targets, metadata)

    # The module creates messages list: [str(d) + "\n" + prompt for d in data]
    expected_messages = [f"Dummy response Nr. {i}" for i in range(len(batch))]
    expected_predictions = model(expected_messages)

    output = language_module.validation_step(batch)

    assert "predictions" in output
    assert "targets" in output
    assert "metadata" in output
    assert output["predictions"] == expected_predictions
    assert output["targets"] == targets
    assert output["metadata"] == metadata


def test_init_attributes(model):
    """Test the attributes of the LanguageModule class."""
    module_instance = LanguageModule(model=model)
    assert module_instance.model is model


class DummyModel(nn.Module):
    """A simple text model for testing purposes."""

    def forward(self, batch: TextBatch) -> list[str]:
        """Generate some text based on the input prompt."""
        return [f"Dummy response Nr. {i}" for i in range(len(batch))]


@pytest.fixture
def model():
    """Return a dummy model instance."""
    return DummyModel()


@pytest.fixture
def language_module(model):
    """Return a LanguageModule instance."""
    return LanguageModule(model=model)
