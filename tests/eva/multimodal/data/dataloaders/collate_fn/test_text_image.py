"""Tests for collate functions."""

import torch
from torchvision import tv_tensors

from eva.language.data.messages import UserMessage
from eva.multimodal.data.dataloaders.collate_fn.text_image import text_image_collate
from eva.multimodal.data.datasets.typings import TextImageSample


def test_text_image_collate_with_targets():
    """Test collating samples with targets."""
    samples = [
        TextImageSample(
            text=[UserMessage(content="Text 1")],
            image=tv_tensors.Image(torch.rand(3, 224, 224)),
            target=torch.tensor(0),
            metadata={"id": 1},
        ),
        TextImageSample(
            text=[UserMessage(content="Text 2")],
            image=tv_tensors.Image(torch.rand(3, 224, 224)),
            target=torch.tensor(1),
            metadata={"id": 2},
        ),
    ]

    batch = text_image_collate(samples)

    assert len(batch.text) == 2
    assert batch.text[0][0].content == "Text 1"
    assert batch.text[1][0].content == "Text 2"
    assert len(batch.image) == 2
    assert batch.target is not None
    assert batch.target.shape == (2,)
    assert torch.equal(batch.target, torch.tensor([0, 1]))
    assert batch.metadata == {"id": [1, 2]}


def test_text_image_collate_without_targets():
    """Test collating samples without targets."""
    samples = [
        TextImageSample(
            text=[UserMessage(content="Text A")],
            image=tv_tensors.Image(torch.rand(3, 224, 224)),
            target=None,
            metadata={"key": "val1"},
        ),
        TextImageSample(
            text=[UserMessage(content="Text B")],
            image=tv_tensors.Image(torch.rand(3, 224, 224)),
            target=None,
            metadata={"key": "val2"},
        ),
    ]

    batch = text_image_collate(samples)

    assert len(batch.text) == 2
    assert len(batch.image) == 2
    assert batch.target is None
    assert batch.metadata == {"key": ["val1", "val2"]}


def test_text_image_collate_without_metadata():
    """Test collating samples without metadata."""
    samples = [
        TextImageSample(
            text=[UserMessage(content="Text")],
            image=tv_tensors.Image(torch.rand(3, 224, 224)),
            target=torch.tensor(0),
            metadata=None,
        ),
        TextImageSample(
            text=[UserMessage(content="Text")],
            image=tv_tensors.Image(torch.rand(3, 224, 224)),
            target=torch.tensor(1),
            metadata=None,
        ),
    ]

    batch = text_image_collate(samples)

    assert len(batch.text) == 2
    assert len(batch.image) == 2
    assert batch.target is not None
    assert batch.target.shape == (2,)
    assert batch.metadata is None
