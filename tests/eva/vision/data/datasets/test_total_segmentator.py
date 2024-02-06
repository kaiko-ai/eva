"""TotalSegmentator dataset tests."""

import os
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split",
    ["train", "val", "test"],
)
def test_sample(total_segmentator_dataset: datasets.TotalSegmentator) -> None:
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
def total_segmentator_dataset(split: Literal["train", "val", "test"], assets_path: str) -> datasets.Bach:
    """TotalSegmentator dataset fixture."""

    with patch("eva.vision.data.datasets.TotalSegmentator._verify_dataset") as _:
        ds = datasets.TotalSegmentator(
            root_dir=os.path.join(assets_path, "vision", "datasets", "total_segmentator"),
            split=split,
            split_ratios=datasets.total_segmentator.SplitRatios(0.33, 0.33, 0.33),
            sample_every_n_slice=1,
        )
        ds.prepare_data()
        ds.setup()
        return ds
