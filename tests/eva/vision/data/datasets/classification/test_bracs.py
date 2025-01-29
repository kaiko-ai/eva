"""BRACS dataset tests."""

import os
from typing import Literal
from unittest import mock

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
        ("val", 1),
        ("test", 0),
        ("test", 1),
    ],
)
def test_sample(bracs_dataset: datasets.BRACS, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = bracs_dataset[index]
    # assert data sample is a tuple
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `target`
    image, target, _ = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, 40, 40)
    assert isinstance(target, torch.Tensor)
    assert target in [0, 1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture(scope="function")
def bracs_dataset(split: Literal["train", "val"], assets_path: str) -> datasets.BRACS:
    """BRACS dataset fixture."""
    with mock.patch.object(
        datasets.BRACS, "classes", new_callable=mock.PropertyMock
    ) as mock_classes:
        mock_classes.return_value = ["0_N", "1_PB"]
        dataset = datasets.BRACS(
            root=os.path.join(assets_path, "vision", "datasets", "bracs"),
            split=split,
        )
        dataset.prepare_data()
        dataset.configure()
        return dataset
