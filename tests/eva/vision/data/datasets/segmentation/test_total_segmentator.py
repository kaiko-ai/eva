"""TotalSegmentator2D dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 9), ("val", 9), (None, 9)],
)
def test_length(
    total_segmentator_dataset: datasets.TotalSegmentator2D, expected_length: int
) -> None:
    """Tests the length of the dataset."""
    assert len(total_segmentator_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
        ("train", 0),
        ("val", 0),
    ],
)
def test_sample(total_segmentator_dataset: datasets.TotalSegmentator2D, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = total_segmentator_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `mask`
    image, mask = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, 16, 16)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (16, 16)


@pytest.fixture(scope="function")
def total_segmentator_dataset(
    split: Literal["train", "val"] | None, assets_path: str
) -> datasets.TotalSegmentator2D:
    """TotalSegmentator2D dataset fixture."""
    dataset = datasets.TotalSegmentator2D(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "total_segmentator",
            "Totalsegmentator_dataset_v201",
        ),
        split=split,
        version=None,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
