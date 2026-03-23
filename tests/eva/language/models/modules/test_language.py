"""Tests the language model module."""

import pytest
from typing_extensions import override

from eva.language.data.messages import UserMessage
from eva.language.models import LanguageModule
from eva.language.models.typings import ModelOutput, TextBatch
from eva.language.models.wrappers.base import LanguageModel


def test_forward(language_module, batch: TextBatch):
    """Test the forward method of the LanguageModule class."""
    expected = ["Dummy response Nr. 0"]
    output = language_module.forward(batch)
    assert output["generated_text"] == expected


def test_validation_step(language_module, batch: TextBatch):
    """Test the validation_step method of the LanguageModule class."""
    # The module creates messages list: [str(d) + "\n" + prompt for d in data]
    expected_predictions = [f"Dummy response Nr. {i}" for i in range(len(batch.text))]

    output = language_module.validation_step(batch)

    assert "predictions" in output
    assert "targets" in output
    assert "metadata" in output
    assert output["predictions"] == expected_predictions
    assert output["targets"] == batch.target
    assert output["metadata"] == batch.metadata


def test_init_attributes(model):
    """Test the attributes of the LanguageModule class."""
    module_instance = LanguageModule(model=model)
    assert module_instance.model is model


class DummyModel(LanguageModel):
    """A simple text model for testing purposes."""

    @override
    def format_inputs(self, batch: TextBatch) -> TextBatch:
        return batch

    @override
    def model_forward(self, batch: TextBatch) -> ModelOutput:
        """Generate text responses based on the batch size."""
        text, _, _ = batch
        return ModelOutput(generated_text=[f"Dummy response Nr. {i}" for i in range(len(text))])


@pytest.fixture
def batch():
    """Return a dummy TextBatch for testing."""
    data = "What is the capital of France?"
    targets = ["Paris"]
    metadata = {"id": [1]}
    return TextBatch(text=[[UserMessage(content=data)]], target=targets, metadata=metadata)


@pytest.fixture
def model():
    """Return a dummy model instance."""
    return DummyModel(system_prompt=None)


@pytest.fixture
def language_module(model):
    """Return a LanguageModule instance."""
    return LanguageModule(model=model)
