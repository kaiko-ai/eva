"""Camelyon16 dataset tests."""

import os
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
import torchvision.transforms.v2 as torch_transforms
from torchvision import tv_tensors

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
    train_dataset = datasets.Camelyon16(root=root, split="train", **DEFAULT_ARGS)
    val_dataset = datasets.Camelyon16(root=root, split="val", **DEFAULT_ARGS)
    test_dataset = datasets.Camelyon16(root=root, split="test", **DEFAULT_ARGS)

    _setup_datasets(train_dataset, val_dataset, test_dataset)

    assert len(train_dataset.datasets) == 3
    assert len(val_dataset.datasets) == 1
    assert len(test_dataset.datasets) == 2

    assert len(train_dataset) == 192
    assert len(val_dataset) == 64
    assert len(test_dataset) == 128

    _check_batch_shape(train_dataset[0])
    _check_batch_shape(val_dataset[0])
    _check_batch_shape(test_dataset[0])


@pytest.mark.parametrize("split", ["train", "val", "test", None])
def test_filenames(root: str, split: Literal["train", "val", "test"]):
    """Tests that the number of filenames matches the dataset size."""
    dataset = datasets.Camelyon16(root=root, split=split, **DEFAULT_ARGS)
    _setup_datasets(dataset)

    filenames = set()
    for i in range(len(dataset)):
        filenames.add(dataset.filename(i))

    assert len(filenames) == len(dataset.datasets)


def _check_batch_shape(batch: Any):
    assert isinstance(batch, tuple)
    assert len(batch) == 3

    image, target, metadata = batch
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, TARGET_SIZE, TARGET_SIZE)

    assert isinstance(target, torch.Tensor)
    assert isinstance(metadata, dict)
    assert "wsi_id" in metadata
    assert "x" in metadata
    assert "y" in metadata
    assert "width" in metadata
    assert "height" in metadata
    assert "level_idx" in metadata


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root directory of the dataset."""
    return os.path.join(assets_path, "vision/datasets/camelyon16")


def _setup_datasets(*datasets: datasets.Camelyon16):
    for dataset in datasets:
        dataset.setup()


@pytest.fixture(autouse=True)
def mock_validate():
    """Mocks the data validation function to avoid expecting all input images."""
    with patch.object(datasets.Camelyon16, "validate", return_value=None):
        yield
