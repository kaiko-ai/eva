"""LiteLLM multimodal wrapper tests."""

import pytest
import torch
from torchvision import tv_tensors

from eva.language.data.messages import UserMessage
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers.litellm import LiteLLMModel

DUMMY_RESPONSE = {"choices": [{"message": {"content": "Test response", "role": "assistant"}}]}


def test_generate_with_image(model_instance, sample_image):
    """Test that the generate method works with image input."""
    batch = TextImageBatch(
        text=[[UserMessage(content="What's in this image?")]],
        image=[sample_image],
        target=None,
        metadata={},
    )
    result = model_instance(batch)
    assert result["generated_text"] == ["Test response"]


def test_generate_without_image(model_instance):
    """Test that the generate method works without image input."""
    # Create a dummy image tensor, but we'll mock the response anyway
    dummy_image = tv_tensors.Image(torch.zeros(3, 1, 1))
    batch = TextImageBatch(
        text=[[UserMessage(content="Hello, world!")]],
        image=[dummy_image],
        target=None,
        metadata={},
    )
    result = model_instance(batch)
    assert result["generated_text"] == ["Test response"]


def test_format_inputs_with_image(model_instance, sample_image):
    """Test format_inputs properly formats messages with images."""
    batch = TextImageBatch(
        text=[[UserMessage(content="Describe this")]],
        image=[sample_image],
        target=None,
        metadata={},
    )
    formatted = model_instance.format_inputs(batch)

    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert len(formatted[0]) == 2  # System message + user message
    assert formatted[0][0]["role"] == "system"
    assert formatted[0][0]["content"] == "You are a helpful assistant."
    assert formatted[0][1]["role"] == "user"
    assert isinstance(formatted[0][1]["content"], list)
    assert len(formatted[0][1]["content"]) == 2
    assert formatted[0][1]["content"][0]["type"] == "text"
    assert formatted[0][1]["content"][1]["type"] == "image_url"


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    return tv_tensors.Image(torch.rand(3, 224, 224))


@pytest.fixture
def fake_completion(monkeypatch):
    """Fixture to override `batch_completion` function and set a dummy OPENAI_API_KEY."""

    def _fake_batch_completion(**_kwargs):
        return [DUMMY_RESPONSE]

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setattr(
        "eva.language.models.wrappers.litellm.batch_completion", _fake_batch_completion
    )
    return _fake_batch_completion


@pytest.fixture
def model_instance(fake_completion):  # noqa: ARG001
    """Fixture to instantiate the multimodal LiteLLMModel."""
    return LiteLLMModel(
        "openai/gpt-4-vision-preview",
        model_kwargs={"temperature": 0.7},
        system_prompt="You are a helpful assistant.",
    )
