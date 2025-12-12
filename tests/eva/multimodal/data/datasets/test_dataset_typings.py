"""Tests for multimodal dataset typings."""

import torch
from torchvision import tv_tensors

from eva.language.data.messages import MessageSeries, UserMessage
from eva.multimodal.data.datasets.typings import TextImageSample


def test_text_image_sample_creation():
    """Test TextImageSample creation and field access."""
    text: MessageSeries = [UserMessage(content="Test message")]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]
    target = 1
    metadata = {"key": "value"}

    sample = TextImageSample(text=text, images=images, target=target, metadata=metadata)

    assert sample.text == text
    assert sample.images is images
    assert sample.target == target
    assert sample.metadata == metadata


def test_text_image_sample_with_none_fields():
    """Test TextImageSample with None target and metadata."""
    text: MessageSeries = [UserMessage(content="Test")]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]

    sample = TextImageSample(text=text, images=images, target=None, metadata=None)

    assert sample.text == text
    assert sample.images is images
    assert sample.target is None
    assert sample.metadata is None


def test_text_image_sample_unpacking():
    """Test TextImageSample can be unpacked."""
    text: MessageSeries = [UserMessage(content="Test")]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]

    sample = TextImageSample(text=text, images=images, target=42, metadata={"test": True})

    unpacked_text, unpacked_images, unpacked_target, unpacked_metadata = sample

    assert unpacked_text == text
    assert unpacked_images is images
    assert unpacked_target == 42
    assert unpacked_metadata == {"test": True}
