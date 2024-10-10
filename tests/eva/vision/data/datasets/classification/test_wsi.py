"""WsiClassificationDataset tests."""

import os
import pickle
import re
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
    "manifest_file": "manifest.csv",
    "width": 32,
    "height": 32,
    "target_mpp": 0.25,
    "sampler": samplers.GridSampler(),
    "backend": "openslide",
    "image_transforms": torch_transforms.Compose([eva_transforms.ResizeAndCrop(size=TARGET_SIZE)]),
}


def test_pickleable(dataset: datasets.WsiClassificationDataset):
    """Tests if the dataset is pickleable (required for multi-worker torch data loaders)."""
    pickled = pickle.dumps(dataset)

    # Check if it works after unpickling
    unpickled_dataset = pickle.loads(pickled)
    for batch in unpickled_dataset:
        _check_batch_shape(batch)


def test_split(root: str):
    """Test loading the dataset with different splits."""
    dataset = datasets.WsiClassificationDataset(root=root, split=None, **DEFAULT_ARGS)
    dataset.setup()
    assert len(dataset) == 192
    _check_batch_shape(dataset[0])

    train_dataset = datasets.WsiClassificationDataset(root=root, split="train", **DEFAULT_ARGS)
    train_dataset.setup()
    assert len(train_dataset) == 64
    _check_batch_shape(train_dataset[0])


def test_filename(dataset: datasets.WsiClassificationDataset):
    """Tests the filename method."""
    pattern = r"^\d+/[a-z]\.tiff$"
    for i in range(len(dataset)):
        assert bool(re.match(pattern, dataset.filename(i)))


def test_missing_columns(root: str):
    """Test if error is raised if columns are missing in the manifest file."""
    with pytest.raises(ValueError, match="Missing columns in the manifest file"):
        datasets.WsiClassificationDataset(
            root=root,
            column_mapping={"target": "label"},
            **DEFAULT_ARGS,
        )


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
    assert "x" in metadata
    assert "y" in metadata
    assert "width" in metadata
    assert "height" in metadata
    assert "level_idx" in metadata


@pytest.fixture
def dataset(root: str) -> datasets.WsiClassificationDataset:
    """Fixture returning a dataset instance."""
    dataset = datasets.WsiClassificationDataset(root=root, **DEFAULT_ARGS)
    dataset.setup()
    return dataset


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root path to the test dataset assets."""
    return os.path.join(assets_path, "vision/datasets/wsi")
