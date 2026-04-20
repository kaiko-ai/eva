"""WORD dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets
from eva.vision.data.tv_tensors import Volume


@pytest.mark.parametrize(
    "split, expected_length",
    [(None, 3), ("train", 2), ("val", 1)],
)
def test_length(word_dataset: datasets.WORD, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(word_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
    ],
)
def test_sample(word_dataset: datasets.WORD, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = word_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, Volume)
    assert image.shape == (20, 1, 64, 64)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (20, 1, 64, 64)
    assert isinstance(metadata, dict)


@pytest.fixture(scope="function")
def word_dataset(split: Literal["train", "val"] | None, assets_path: str) -> datasets.WORD:
    """WORD dataset fixture."""
    dataset = datasets.WORD(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "word",
        ),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
