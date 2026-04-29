"""CC-CCII dataset tests."""

import os
from typing import Literal

import pytest
import torch

from eva.vision.data import datasets
from eva.vision.data.tv_tensors import Volume


@pytest.mark.parametrize(
    "split, index",
    [
        ("train", 0),
        ("val", 0),
    ],
)
def test_sample(cc_ccii_dataset: datasets.CC_CCII, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = cc_ccii_dataset[index]
    # assert data sample is a tuple
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `target`
    image, target, _ = sample
    assert isinstance(image, Volume)
    assert image.shape == (6, 1, 96, 96)
    assert isinstance(target, torch.Tensor)
    assert target in [0, 1]


@pytest.fixture(scope="function")
def cc_ccii_dataset(split: Literal["train", "val"], assets_path: str) -> datasets.CC_CCII:
    """CC-CCII dataset fixture."""
    dataset = datasets.CC_CCII(
        root=os.path.join(assets_path, "vision", "datasets", "cc_ccii"),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
