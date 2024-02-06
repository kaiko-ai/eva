"""BACH dataset tests."""

import os
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 16), ("val", 4), ("test", 4)],
)
def test_length(bach_dataset: datasets.Bach, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(bach_dataset) == expected_length


@pytest.mark.parametrize(
    "split",
    ["train", "val", "test"],
)
def test_sample(bach_dataset: datasets.Bach) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = bach_dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    image, target = sample
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16, 3)
    assert isinstance(target, np.ndarray)
    assert target in [0, 1, 2, 3]


@pytest.fixture(scope="function")
def bach_dataset(split: Literal["train", "val", "test"], assets_path: str) -> datasets.Bach:
    """BACH dataset fixture."""
    with patch("eva.vision.data.datasets.Bach._verify_dataset") as _:
        ds = datasets.Bach(
            root_dir=os.path.join(assets_path, "vision", "datasets", "bach"),
            split=split,
        )
        ds.setup()
        return ds
