"""WsiClassificationDataset tests."""

import os
import pickle
from typing import Any

import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as torch_transforms

from eva.vision.data import datasets
from eva.vision.data import transforms as eva_transforms
from eva.vision.data.wsi.patching import samplers

DEFAULT_ARGS = {
    "manifest_file": "manifest.csv",
    "width": 224,
    "height": 224,
    "target_mpp": 0.25,
    "sampler": samplers.GridSampler(10),
    "backend": "openslide",
    "image_transforms": torch_transforms.Compose([eva_transforms.ResizeAndCrop(size=224)]),
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
    assert len(dataset) == 3
    _check_batch_shape(dataset[0])

    train_dataset = datasets.WsiClassificationDataset(root=root, split="train", **DEFAULT_ARGS)
    assert len(train_dataset) == 1
    _check_batch_shape(train_dataset[0])


def test_filename(dataset: datasets.WsiClassificationDataset):
    """Tests the filename method."""
    assert dataset.filename(0) == "0/a.tiff"
    assert dataset.filename(1) == "0/b.tiff"
    assert dataset.filename(2) == "1/a.tiff"


def _check_batch_shape(batch: Any):
    assert isinstance(batch, tuple)
    assert len(batch) == 2

    image, target = batch
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)

    assert isinstance(target, np.ndarray)
    assert target.size == 1


@pytest.fixture
def dataset(root: str) -> datasets.WsiClassificationDataset:
    """Fixture returning a dataset instance."""
    return datasets.WsiClassificationDataset(root=root, **DEFAULT_ARGS)


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root path to the test dataset assets."""
    return os.path.join(assets_path, "vision/datasets/wsi")
