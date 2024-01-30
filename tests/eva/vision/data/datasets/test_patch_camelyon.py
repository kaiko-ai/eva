"""Patch Camelyon dataset tests."""

import os
from typing import Literal

import numpy as np
import pytest

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 4), ("valid", 2), ("test", 1)],
)
def test_length(pcam_dataset: datasets.PatchCamelyon, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(pcam_dataset) == expected_length


@pytest.mark.parametrize(
    "split",
    ["train", "valid", "test"],
)
def test_sample(pcam_dataset: datasets.PatchCamelyon) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = pcam_dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    image, target = sample
    assert isinstance(image, np.ndarray)
    assert image.shape == (96, 96, 3)
    assert isinstance(target, np.ndarray)
    assert target in [0, 1]


@pytest.fixture(scope="function")
def pcam_dataset(
    split: Literal["train", "valid", "test"], assets_path: str
) -> datasets.PatchCamelyon:
    """PatchCamelyon dataset fixture."""
    return datasets.PatchCamelyon(
        root=os.path.join(assets_path, "vision", "datasets", "pcam"),
        split=split,
    )
