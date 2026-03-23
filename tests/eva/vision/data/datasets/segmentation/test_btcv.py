"""BTCV dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets
from eva.vision.data import tv_tensors as eva_tv_tensors


@pytest.mark.parametrize(
    "split, expected_length",
    [(None, 2), ("train", 1), ("val", 1)],
)
def test_length(btcv_dataset: datasets.BTCV, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(btcv_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
    ],
)
def test_sample(btcv_dataset: datasets.BTCV, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = btcv_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, eva_tv_tensors.Volume)
    assert image.shape == (4, 1, 8, 8)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (4, 1, 8, 8)
    assert isinstance(metadata, dict)


@pytest.fixture(scope="function")
def btcv_dataset(split: Literal["train", "val"] | None, assets_path: str) -> datasets.BTCV:
    """BTCV dataset fixture."""
    dataset = datasets.BTCV(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "btcv",
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
