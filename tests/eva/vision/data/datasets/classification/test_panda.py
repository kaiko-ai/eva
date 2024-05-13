"""PANDA dataset tests."""

import os
from typing import Any

import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as torch_transforms

from eva.vision.data import datasets
from eva.vision.data import transforms as eva_transforms
from eva.vision.data.wsi.patching import samplers

TARGET_SIZE = 224
DEFAULT_ARGS = {
    "width": 16,
    "height": 16,
    "target_mpp": 0.5,
    "sampler": samplers.GridSampler(),
    "backend": "openslide",
    "image_transforms": torch_transforms.Compose([eva_transforms.ResizeAndCrop(size=TARGET_SIZE)]),
}


def test_split_and_expected_shapes(root: str):
    """Test loading the dataset with different splits."""
    train_dataset = datasets.PANDA(root=root, split="train", **DEFAULT_ARGS)
    val_dataset = datasets.PANDA(root=root, split="val", **DEFAULT_ARGS)
    test_dataset = datasets.PANDA(root=root, split="test", **DEFAULT_ARGS)

    assert len(train_dataset.datasets) == 6
    assert len(val_dataset.datasets) == 2
    assert len(test_dataset.datasets) == 2

    assert len(train_dataset) == 384
    assert len(val_dataset) == 128
    assert len(test_dataset) == 128

    _check_batch_shape(train_dataset[0])
    _check_batch_shape(val_dataset[0])
    _check_batch_shape(test_dataset[0])


def test_same_split_same_seed(root: str):
    """Test that the generated split is deterministic when using the same seed."""
    dataset1 = datasets.PANDA(root=root, split="train", seed=42, **DEFAULT_ARGS)
    dataset2 = datasets.PANDA(root=root, split="train", seed=42, **DEFAULT_ARGS)

    assert len(dataset1) == len(dataset2)
    assert dataset1._file_paths == dataset2._file_paths

    for i in range(len(dataset1)):
        assert np.allclose(dataset1[i][1], dataset2[i][1])


def test_different_seed_different_split(root: str):
    """Test that the generated split is different when using a different seed."""
    dataset1 = datasets.PANDA(root=root, split="train", seed=42, **DEFAULT_ARGS)
    dataset2 = datasets.PANDA(root=root, split="train", seed=43, **DEFAULT_ARGS)

    assert len(dataset1) == len(dataset2)
    assert dataset1._file_paths != dataset2._file_paths


def _check_batch_shape(batch: Any):
    assert isinstance(batch, tuple)
    assert len(batch) == 3

    image, target, metadata = batch
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, TARGET_SIZE, TARGET_SIZE)

    assert isinstance(target, np.ndarray)
    assert target.size == 1

    assert isinstance(metadata, dict)
    assert "wsi_id" in metadata


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root directory of the dataset."""
    return os.path.join(assets_path, "vision/datasets/panda")
