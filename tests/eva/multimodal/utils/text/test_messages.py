"""Tests for message formatting utilities."""

import pytest
import torch
from torchvision import tv_tensors

from eva.language.data.messages import MessageSeries, SystemMessage, UserMessage
from eva.multimodal.utils.text.messages import format_huggingface_message, format_litellm_message


def test_format_huggingface_message_without_images():
    """Test formatting messages for HuggingFace without images."""
    messages: MessageSeries = [UserMessage(content="Hello")]
    formatted = format_huggingface_message(messages, images=None)

    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"


def test_format_huggingface_message_with_images():
    """Test formatting messages for HuggingFace with images."""
    messages: MessageSeries = [
        SystemMessage(content="System prompt"),
        UserMessage(content="What's this?"),
    ]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]
    formatted = format_huggingface_message(messages, images=images)

    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == "System prompt"
    assert formatted[1]["role"] == "user"
    assert isinstance(formatted[1]["content"], list)
    assert formatted[1]["content"][0]["type"] == "text"
    assert formatted[1]["content"][1]["type"] == "image"


def test_format_litellm_message_without_image():
    """Test formatting messages for LiteLLM without image."""
    messages: MessageSeries = [UserMessage(content="Hello")]
    formatted = format_litellm_message(messages, images=None)

    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"


@pytest.mark.parametrize(
    "image_format,expected_prefix",
    [
        ("jpeg", "data:image/jpeg;base64,"),
        ("png", "data:image/png;base64,"),
    ],
)
def test_format_litellm_message_with_image(image_format, expected_prefix):
    """Test formatting messages for LiteLLM with different image formats."""
    messages: MessageSeries = [
        SystemMessage(content="System prompt"),
        UserMessage(content="Describe this"),
    ]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]
    formatted = format_litellm_message(messages, images=images, image_format=image_format)

    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == "System prompt"
    assert formatted[1]["role"] == "user"
    assert isinstance(formatted[1]["content"], list)
    assert formatted[1]["content"][0]["type"] == "text"
    assert formatted[1]["content"][0]["text"] == "Describe this"
    assert formatted[1]["content"][1]["type"] == "image_url"
    assert "url" in formatted[1]["content"][1]["image_url"]
    assert formatted[1]["content"][1]["image_url"]["url"].startswith(expected_prefix)
