"""Tiger WSIBULK dataset tests."""

import os
from typing import Any, Literal

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
    "sampler": samplers.GridSampler(),
    "backend": "openslide",
    "image_transforms": torch_transforms.Compose([eva_transforms.ResizeAndCrop(size=TARGET_SIZE)]),
    "embeddings_dir": "tests/eva/assets/vision/datasets/tiger_wsibulk/embeddings_dir",
}


def test_split_and_expected_shapes(root: str, monkeypatch):
    """Test loading the dataset with different splits."""

    train_dataset = datasets.TIGERWsiBulk(root=root, split="train", **DEFAULT_ARGS)
    val_dataset = datasets.TIGERWsiBulk(root=root, split="val", **DEFAULT_ARGS)
    test_dataset = datasets.TIGERWsiBulk(root=root, split="test", **DEFAULT_ARGS)

    _setup_datasets(train_dataset, val_dataset, test_dataset, monkeypatch=monkeypatch)

    assert len(train_dataset) == 192
    assert len(val_dataset) == 64
    assert len(test_dataset) == 64

    _check_batch_shape(train_dataset[0])
    _check_batch_shape(val_dataset[0])
    _check_batch_shape(test_dataset[0])


@pytest.mark.parametrize("split", ["train", "val", "test", None])
def test_filenames(root: str, split: Literal["train", "val", "test"], monkeypatch):
    """Tests that the number of filenames matches the dataset size."""
    dataset = datasets.TIGERWsiBulk(root=root, split=split, **DEFAULT_ARGS)
    _setup_datasets(dataset, monkeypatch=monkeypatch)

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
    return os.path.join(assets_path, "vision/datasets/tiger_wsibulk")


def _setup_datasets(*dataset_splits: datasets.TIGERWsiBulk, monkeypatch):

    monkeypatch.setattr(
        datasets.TIGERWsiBulk,
        "_expected_dataset_lengths",
        {"train": 3, "val": 1, "test": 1, None: 5},
    )

    split_to_file = {
        "train": "coords_train.csv",
        "val": "coords_val.csv",
        "test": "coords_test.csv",
    }

    for dataset in dataset_splits:

        split = dataset._split
        if split is not None:
            csv_file = split_to_file[split]
            monkeypatch.setattr(
                dataset,
                "_coords_path",
                f"tests/eva/assets/vision/datasets/tiger_wsibulk/embeddings_dir/{csv_file}",
            )

        dataset.setup()
