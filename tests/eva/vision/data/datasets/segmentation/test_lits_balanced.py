"""LiTS dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [(None, 4)],
)
def test_length(lits_balanced_dataset: datasets.LiTSBalanced, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(lits_balanced_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
    ],
)
def test_sample(lits_balanced_dataset: datasets.LiTSBalanced, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = lits_balanced_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (1, 512, 512)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (512, 512)
    assert isinstance(metadata, dict)
    assert "slice_index" in metadata


@pytest.fixture(scope="function")
def lits_balanced_dataset(
    split: Literal["train", "val", "test"] | None, assets_path: str
) -> datasets.LiTSBalanced:
    """LiTS dataset fixture."""
    dataset = datasets.LiTSBalanced(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "lits",
        ),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
