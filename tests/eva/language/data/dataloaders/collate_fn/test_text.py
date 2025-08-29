"""Tests for text collate functions."""

import torch

from eva.language.data.dataloaders.collate_fn.text import text_collate
from eva.language.data.datasets.typings import TextSample
from eva.language.data.messages import UserMessage


def test_text_collate_with_targets():
    """Test collating samples with targets."""
    samples = [
        TextSample(
            text=[UserMessage(content="Text 1")],
            target=torch.tensor(0),
            metadata={"id": 1, "category": "A"},
        ),
        TextSample(
            text=[UserMessage(content="Text 2")],
            target=torch.tensor(1),
            metadata={"id": 2, "category": "B"},
        ),
        TextSample(
            text=[UserMessage(content="Text 3")],
            target=torch.tensor(2),
            metadata={"id": 3, "category": "A"},
        ),
    ]

    batch = text_collate(samples)

    assert len(batch.text) == 3
    assert batch.text[0][0].content == "Text 1"
    assert batch.text[1][0].content == "Text 2"
    assert batch.text[2][0].content == "Text 3"
    assert batch.target is not None
    assert batch.target.shape == (3,)
    assert torch.equal(batch.target, torch.tensor([0, 1, 2]))
    assert batch.metadata == {"id": [1, 2, 3], "category": ["A", "B", "A"]}


def test_text_collate_without_targets():
    """Test collating samples without targets."""
    samples = [
        TextSample(
            text=[UserMessage(content="Text A")],
            target=None,
            metadata={"key": "val1", "score": 0.5},
        ),
        TextSample(
            text=[UserMessage(content="Text B")],
            target=None,
            metadata={"key": "val2", "score": 0.8},
        ),
    ]

    batch = text_collate(samples)

    assert len(batch.text) == 2
    assert batch.text[0][0].content == "Text A"
    assert batch.text[1][0].content == "Text B"
    assert batch.target is None
    assert batch.metadata == {"key": ["val1", "val2"], "score": [0.5, 0.8]}


def test_text_collate_without_metadata():
    """Test collating samples without metadata."""
    samples = [
        TextSample(
            text=[UserMessage(content="Text 1")],
            target=torch.tensor(0),
            metadata=None,
        ),
        TextSample(
            text=[UserMessage(content="Text 2")],
            target=torch.tensor(1),
            metadata=None,
        ),
    ]

    batch = text_collate(samples)

    assert len(batch.text) == 2
    assert batch.text[0][0].content == "Text 1"
    assert batch.text[1][0].content == "Text 2"
    assert batch.target is not None
    assert batch.target.shape == (2,)
    assert torch.equal(batch.target, torch.tensor([0, 1]))
    assert batch.metadata is None


def test_text_collate_with_multiple_messages():
    """Test collating samples with multiple messages in conversation."""
    samples = [
        TextSample(
            text=[
                UserMessage(content="Question 1"),
                UserMessage(content="Follow-up 1"),
            ],
            target=torch.tensor([1, 0, 0]),
            metadata={"sample_id": "s1"},
        ),
        TextSample(
            text=[
                UserMessage(content="Question 2"),
                UserMessage(content="Follow-up 2"),
            ],
            target=torch.tensor([0, 1, 0]),
            metadata={"sample_id": "s2"},
        ),
    ]

    batch = text_collate(samples)

    assert len(batch.text) == 2
    assert len(batch.text[0]) == 2
    assert len(batch.text[1]) == 2
    assert batch.text[0][0].content == "Question 1"
    assert batch.text[0][1].content == "Follow-up 1"
    assert batch.text[1][0].content == "Question 2"
    assert batch.text[1][1].content == "Follow-up 2"
    assert batch.target is not None
    assert batch.target.shape == (2, 3)
    assert torch.equal(batch.target, torch.tensor([[1, 0, 0], [0, 1, 0]]))
    assert batch.metadata == {"sample_id": ["s1", "s2"]}


def test_text_collate_with_mixed_metadata():
    """Test collating samples where some have metadata and some don't."""
    samples = [
        TextSample(
            text=[UserMessage(content="Text with metadata")],
            target=torch.tensor(0.5),
            metadata={"has_meta": True},
        ),
        TextSample(
            text=[UserMessage(content="Text without metadata")],
            target=torch.tensor(0.7),
            metadata=None,
        ),
    ]

    batch = text_collate(samples)

    assert len(batch.text) == 2
    assert batch.text[0][0].content == "Text with metadata"
    assert batch.text[1][0].content == "Text without metadata"
    assert batch.target is not None
    assert batch.target.shape == (2,)
    assert torch.allclose(batch.target, torch.tensor([0.5, 0.7]))
    assert batch.metadata == {"has_meta": [True]}


def test_text_collate_empty_batch():
    """Test collating an empty batch."""
    samples = []

    try:
        _ = text_collate(samples)
        raise AssertionError("Should raise an error for empty batch")
    except (ValueError, IndexError):
        pass


def test_text_collate_single_sample():
    """Test collating a single sample."""
    samples = [
        TextSample(
            text=[UserMessage(content="Single text")],
            target=torch.tensor([1.0, 2.0, 3.0]),
            metadata={"single": True},
        )
    ]

    batch = text_collate(samples)

    assert len(batch.text) == 1
    assert batch.text[0][0].content == "Single text"
    assert batch.target is not None
    assert batch.target.shape == (1, 3)
    assert torch.equal(batch.target, torch.tensor([[1.0, 2.0, 3.0]]))
    assert batch.metadata == {"single": [True]}
