"""Tests for multimodal model typings."""

import torch
from torchvision import tv_tensors

from eva.language.data.messages import MessageSeries, UserMessage
from eva.multimodal.models.typings import TextImageBatch


def test_text_image_batch_creation():
    """Test TextImageBatch creation and field access."""
    messages: list[MessageSeries] = [[UserMessage(content="Test")]]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]
    target = torch.tensor([1])
    metadata = {"key": "value"}

    batch = TextImageBatch(text=messages, image=images, target=target, metadata=metadata)

    assert batch.text == messages
    assert batch.image == images
    assert batch.target is not None and torch.equal(batch.target, target)
    assert batch.metadata == metadata


def test_text_image_batch_unpacking():
    """Test TextImageBatch can be unpacked."""
    messages: list[MessageSeries] = [[UserMessage(content="Test")]]
    images = [tv_tensors.Image(torch.rand(3, 224, 224))]

    batch = TextImageBatch(text=messages, image=images, target=None, metadata=None)

    text, image, target, metadata = batch
    assert text == messages
    assert image == images
    assert target is None
    assert metadata is None
