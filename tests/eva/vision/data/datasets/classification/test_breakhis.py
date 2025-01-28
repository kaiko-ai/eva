"""BreaKHis dataset tests."""

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
        ("train", 1),
        ("val", 0),
    ],
)
def test_sample(breakhis_dataset: datasets.BreaKHis, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = breakhis_dataset[index]
    # assert data sample is a tuple
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `target`
    image, target, _ = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, 40, 40)
    assert isinstance(target, torch.Tensor)
    assert target in [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.fixture(scope="function")
def breakhis_dataset(split: Literal["train", "val"], assets_path: str) -> datasets.BreaKHis:
    """BreaKHis dataset fixture."""
    dataset = datasets.BreaKHis(
        root=os.path.join(assets_path, "vision", "datasets", "breakhis"),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()

    return dataset
