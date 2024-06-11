"""CRC dataset tests."""

import os
from typing import Literal

import pytest
import torch
from torchvision import tv_tensors

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, index",
    [
        ("train", 0),
        ("train", 2),
        ("val", 0),
        ("val", 2),
    ],
)
def test_sample(crc_dataset: datasets.CRC, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = crc_dataset[index]
    # assert data sample is a tuple
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `target`
    image, target, _ = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, 16, 16)
    assert isinstance(target, torch.Tensor)
    assert target in [0, 1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture(scope="function")
def crc_dataset(split: Literal["train", "val"], assets_path: str) -> datasets.CRC:
    """CRC dataset fixture."""
    dataset = datasets.CRC(
        root=os.path.join(assets_path, "vision", "datasets", "crc"),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
