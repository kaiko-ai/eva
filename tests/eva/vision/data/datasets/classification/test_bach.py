"""BACH dataset tests."""

import os
from typing import Literal

import numpy as np
import pytest

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 268), ("val", 132), (None, 400)],
)
def test_length(bach_dataset: datasets.BACH, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(bach_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
        (None, 9),
        ("train", 0),
        ("train", 2),
    ],
)
def test_sample(bach_dataset: datasets.BACH, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = bach_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    image, target = sample
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16, 3)
    assert isinstance(target, np.ndarray)
    assert target in [0, 1, 2, 3]


@pytest.fixture(scope="function")
def bach_dataset(split: Literal["train", "val"], assets_path: str) -> datasets.BACH:
    """BACH dataset fixture."""
    dataset = datasets.BACH(
        root=os.path.join(assets_path, "vision", "datasets", "bach"),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
