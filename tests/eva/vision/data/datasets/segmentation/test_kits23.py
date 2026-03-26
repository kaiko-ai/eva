"""KiTS23 dataset tests."""

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
def test_length(kits23_dataset: datasets.KiTS23, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(kits23_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
    ],
)
def test_sample(kits23_dataset: datasets.KiTS23, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = kits23_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, Volume)
    assert image.shape == (77, 1, 64, 64)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (77, 1, 64, 64)
    assert isinstance(metadata, dict)


@pytest.fixture(scope="function")
def kits23_dataset(split: Literal["train", "val"] | None, assets_path: str) -> datasets.KiTS23:
    """kits23 dataset fixture."""
    dataset = datasets.KiTS23(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "kits23",
        ),
        split=split,
    )
    dataset._train_index_ranges = [(0, 2)]
    dataset._val_index_ranges = [(2, 3)]
    dataset.prepare_data()
    dataset.configure()
    return dataset
