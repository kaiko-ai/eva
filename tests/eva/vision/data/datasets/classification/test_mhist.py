"""MHIST dataset tests."""

import os
from typing import Literal

import pytest
import torch
from torchvision import tv_tensors

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 5), ("test", 2)],
)
def test_length(mhist_dataset: datasets.BACH, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(mhist_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        ("train", 0),
        ("train", 4),
        ("test", 0),
        ("test", 1),
    ],
)
def test_sample(mhist_dataset: datasets.MHIST, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = mhist_dataset[index]
    # assert data sample is a tuple
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `target`
    image, target, _ = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, 224, 224)
    assert isinstance(target, torch.Tensor)
    assert target in [0, 1]


@pytest.fixture(scope="function")
def mhist_dataset(split: Literal["train", "test"], assets_path: str) -> datasets.MHIST:
    """MHIST dataset fixture."""
    dataset = datasets.MHIST(
        root=os.path.join(assets_path, "vision", "datasets", "mhist"),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
