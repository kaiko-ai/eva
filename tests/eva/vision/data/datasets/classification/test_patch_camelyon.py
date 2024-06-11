"""PatchCamelyon dataset tests."""

import os
from typing import Literal

import pytest
import torch
from torchvision import tv_tensors

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 4), ("valid", 2), ("test", 1)],
)
def test_length(patch_camelyon_dataset: datasets.PatchCamelyon, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(patch_camelyon_dataset) == expected_length


@pytest.mark.parametrize(
    "split",
    ["train", "valid", "test"],
)
def test_sample(patch_camelyon_dataset: datasets.PatchCamelyon) -> None:
    """Tests the format of a dataset sample."""
    sample = patch_camelyon_dataset[0]
    # assert data sample is a tuple
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `target`
    image, target, _ = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, 96, 96)
    assert isinstance(target, torch.Tensor)
    assert target in [0, 1]


@pytest.fixture(scope="function")
def patch_camelyon_dataset(
    split: Literal["train", "val", "test"], assets_path: str
) -> datasets.PatchCamelyon:
    """PatchCamelyon dataset fixture."""
    dataset = datasets.PatchCamelyon(
        root=os.path.join(assets_path, "vision", "datasets", "patch_camelyon"),
        split=split,
    )
    return dataset
