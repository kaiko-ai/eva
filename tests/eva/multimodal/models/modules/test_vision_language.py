"""Tests the vision-language model module."""

import pytest
import torch
from torch import nn
from torchvision import tv_tensors

from eva.language.data.messages import MessageSeries, UserMessage
from eva.multimodal.models.modules.vision_language import VisionLanguageModule
from eva.multimodal.models.typings import TextImageBatch


def test_forward(vision_language_module):
    """Test the forward method of the VisionLanguageModule class."""
    text: list[MessageSeries] = [[UserMessage(content="Hello world")]]
    batch = TextImageBatch(
        text=text,
        image=[tv_tensors.Image(torch.rand(3, 224, 224))],
        target=torch.tensor([0]),
        metadata={"id": [1]},
    )
    expected = ["Dummy response Nr. 0"]
    result = vision_language_module.forward(batch)
    assert result == expected


def test_validation_step(vision_language_module):
    """Test the validation_step method of the VisionLanguageModule class."""
    text: list[MessageSeries] = [[UserMessage(content="What is in the image?")]]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]
    targets = torch.tensor([1])
    metadata = {"id": [1], "category": ["test"]}
    batch = TextImageBatch(text=text, image=images, target=targets, metadata=metadata)

    output = vision_language_module.validation_step(batch)

    assert "inputs" in output
    assert "predictions" in output
    assert "targets" in output
    assert "metadata" in output
    assert output["inputs"] == text
    assert output["predictions"] == ["Dummy response Nr. 0"]
    assert torch.equal(output["targets"], targets)
    assert output["metadata"] == metadata


def test_test_step(vision_language_module):
    """Test the test_step method of the VisionLanguageModule class."""
    text: list[MessageSeries] = [
        [UserMessage(content="Describe this image")],
        [UserMessage(content="What do you see?")],
    ]
    images = [
        tv_tensors.Image(torch.rand(3, 224, 224)),
        tv_tensors.Image(torch.rand(3, 224, 224)),
    ]
    targets = torch.tensor([0, 1])
    metadata = {"id": [1, 2]}
    batch = TextImageBatch(text=text, image=images, target=targets, metadata=metadata)

    output = vision_language_module.test_step(batch)

    assert "inputs" in output
    assert "predictions" in output
    assert "targets" in output
    assert "metadata" in output
    assert output["inputs"] == text
    assert output["predictions"] == ["Dummy response Nr. 0", "Dummy response Nr. 1"]
    assert torch.equal(output["targets"], targets)
    assert output["metadata"] == metadata


def test_init_attributes(model):
    """Test the attributes of the VisionLanguageModule class."""
    module_instance = VisionLanguageModule(model=model)
    assert module_instance.model is model
    assert module_instance.metrics is not None  # MetricModule is created by default
    assert module_instance.postprocess is not None  # BatchPostProcess is created by default


def test_batch_step_without_targets(vision_language_module):
    """Test the _batch_step method with None targets."""
    text: list[MessageSeries] = [[UserMessage(content="Test message")]]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]
    batch = TextImageBatch(text=text, image=images, target=None, metadata=None)

    output = vision_language_module.validation_step(batch)

    assert output["targets"] is None
    assert output["metadata"] is None
    assert output["inputs"] == text
    assert output["predictions"] == ["Dummy response Nr. 0"]


class DummyVisionLanguageModel(nn.Module):
    """A simple vision-language model for testing purposes."""

    def forward(self, batch: TextImageBatch) -> list[str]:
        """Generate text responses based on the batch size."""
        text, images, _, _ = batch
        return [f"Dummy response Nr. {i}" for i in range(len(text))]


@pytest.fixture
def model():
    """Return a dummy model instance."""
    return DummyVisionLanguageModel()


@pytest.fixture
def vision_language_module(model):
    """Return a VisionLanguageModule instance."""
    return VisionLanguageModule(model=model)
