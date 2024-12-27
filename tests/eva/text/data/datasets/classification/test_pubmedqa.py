"""PubMedQA dataset tests."""

from typing import Literal

import pytest

from eva.language.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 450), ("test", 500), ("validation", 50), ("train+test+validation", 1000)],
)
def test_length(pubmedqa_dataset: datasets.PubMedQA, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(pubmedqa_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        ("train", 0),
        ("train", 10),
        ("test", 0),
        ("validation", 0),
        ("train+test+validation", 0),
    ],
)
def test_sample(pubmedqa_dataset: datasets.PubMedQA, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = pubmedqa_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3

    text, target, metadata = sample
    assert isinstance(text, str)
    assert text.startswith("Question: ")
    assert "Context: " in text

    assert isinstance(target, int)
    assert target in [0, 1, 2]

    assert isinstance(metadata, dict)
    required_keys = {
        "year",
        "labels",
        "meshes",
        "long_answer",
        "reasoning_required",
        "reasoning_free",
    }
    assert all(key in metadata for key in required_keys)


@pytest.mark.parametrize("split", ["train", "test", "validation", "train+test+validation"])
def test_classes(pubmedqa_dataset: datasets.PubMedQA) -> None:
    """Tests the dataset classes."""
    assert pubmedqa_dataset.classes == ["no", "yes", "maybe"]
    assert pubmedqa_dataset.class_to_idx == {"no": 0, "yes": 1, "maybe": 2}


@pytest.fixture(scope="function")
def pubmedqa_dataset(
    split: Literal["train", "test", "validation", "train+test+validation"]
) -> datasets.PubMedQA:
    """PubMedQA dataset fixture."""
    dataset = datasets.PubMedQA(split=split)
    return dataset
