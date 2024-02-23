"""CRC_HE dataset tests."""

import os
from typing import Literal

import numpy as np
import pytest

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
        (None, 5),
        ("train", 0),
        ("train", 2),
    ],
)
def test_sample(crc_he_dataset: datasets.CRC_HE, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = crc_he_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    image, target = sample
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16, 3)
    assert isinstance(target, np.ndarray)
    assert target in [0, 1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture(scope="function")
def crc_he_dataset(split: Literal["train", "val"], assets_path: str) -> datasets.CRC_HE:
    """CRC_HE dataset fixture."""
    dataset = datasets.CRC_HE(
        root=os.path.join(assets_path, "vision", "datasets", "crc_he"),
        split=split,
    )
    dataset.prepare_data()
    dataset.setup()
    return dataset
