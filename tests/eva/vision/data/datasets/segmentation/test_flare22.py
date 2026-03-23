"""Flare22 dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets
from eva.vision.data.tv_tensors import Volume


@pytest.mark.parametrize(
    "split, expected_length",
    [(None, 2), ("train", 1), ("val", 1)],
)
def test_length(flare22_dataset: datasets.FLARE22, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(flare22_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
    ],
)
def test_sample(flare22_dataset: datasets.FLARE22, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = flare22_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, Volume)
    assert image.shape == (9, 1, 64, 64)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (9, 1, 64, 64)
    assert isinstance(metadata, dict)


@pytest.fixture(scope="function")
def flare22_dataset(split: Literal["train", "val"] | None, assets_path: str) -> datasets.FLARE22:
    """flare22 dataset fixture."""
    dataset = datasets.FLARE22(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "flare22",
        ),
        split=split,
    )
    dataset._split_index_ranges = {
        "train": [(0, 1)],
        "val": [(1, 2)],
        None: [(0, 2)],
    }
    dataset.prepare_data()
    dataset.configure()
    return dataset
