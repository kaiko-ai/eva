"""TotalSegmentator dataset tests."""

import os
from typing import Literal

import numpy as np
import pytest

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split",
    ["train", "val", "test"],
)
def test_sample(total_segmentator_dataset: datasets.TotalSegmentatorClassification) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = total_segmentator_dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    image, target = sample
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16)
    assert isinstance(target, np.ndarray)
    assert all(target == [0, 0, 0])


@pytest.fixture(scope="function")
def total_segmentator_dataset(
    split: Literal["train", "val", "test"],
    assets_path: str,
) -> datasets.TotalSegmentatorClassification:
    """TotalSegmentator dataset fixture."""
    dataset = datasets.TotalSegmentatorClassification(
        root=os.path.join(assets_path, "vision", "datasets", "total_segmentator"),
        split=split,
        n_slices_per_image=2,
    )
    dataset.prepare_data()
    dataset.setup()
    return dataset
