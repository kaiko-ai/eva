"""Tests for the embedding datasets."""

import os

import numpy as np
import pytest
import torch

from eva.vision.data.datasets.embeddings import EmbeddingClassificationDataset, EmbeddingDataset
from eva.vision.data.datasets.typings import DatasetType


@pytest.fixture()
def patch_level_manifest_path(assets_path: str) -> str:
    """Path to a fake patch level manifest."""
    return os.path.join(assets_path, "manifests", "embeddings", "patch_level.csv")


@pytest.fixture()
def slide_level_manifest_path(assets_path: str) -> str:
    """Path to a fake patch level manifest."""
    return os.path.join(assets_path, "manifests", "embeddings", "slide_level.csv")


def test_patch_level_dataset(patch_level_manifest_path: str, assets_path: str):
    """Test that the patch level dataset has the correct length and item shapes/types."""
    ds = EmbeddingDataset(
        manifest_path=patch_level_manifest_path,
        root_dir=assets_path,
        dataset_type=DatasetType.PATCH,
        split=None,
    )
    ds.setup()

    expected_shape = (8,)
    assert len(ds) == 5
    for i in range(len(ds)):
        assert isinstance(ds[i], torch.Tensor)
        assert ds[i].shape == expected_shape


def test_patch_level_classification_dataset(patch_level_manifest_path: str, assets_path: str):
    """Test that the patch level dataset has the correct length and item shapes/types."""
    ds = EmbeddingClassificationDataset(
        manifest_path=patch_level_manifest_path,
        root_dir=assets_path,
        dataset_type=DatasetType.PATCH,
        split=None,
    )
    ds.setup()

    expected_shape = (8,)
    assert len(ds) == 5
    for i in range(len(ds)):
        assert isinstance(ds[i], tuple)
        assert len(ds[i]) == 2
        embedding, target = ds[i]
        assert embedding.shape == expected_shape
        assert np.issubdtype(type(target), int)


def test_slide_level_dataset_length_and_embedding_shape(
    slide_level_manifest_path: str, assets_path: str
):
    """Test that the slide level dataset has the correct length and item shapes/types."""
    ds = EmbeddingDataset(
        manifest_path=slide_level_manifest_path,
        root_dir=assets_path,
        dataset_type=DatasetType.SLIDE,
        split=None,
        n_patches_per_slide=10,
    )
    ds.setup()

    expected_shape = (10, 8)
    assert len(ds) == 3
    for i in range(len(ds)):
        assert ds[i].shape == expected_shape


def test_slide_level_classification_dataset_length_and_embedding_shape(
    slide_level_manifest_path: str, assets_path: str
):
    """Test that the slide level dataset has the correct length and embedding tensor shapes."""
    ds = EmbeddingClassificationDataset(
        manifest_path=slide_level_manifest_path,
        root_dir=assets_path,
        dataset_type=DatasetType.SLIDE,
        split=None,
        n_patches_per_slide=10,
    )
    ds.setup()

    expected_shape = (10, 8)
    assert len(ds) == 3
    for i in range(len(ds)):
        assert isinstance(ds[i], tuple)
        assert len(ds[i]) == 2
        embedding, target = ds[i]
        assert embedding.shape == expected_shape
        assert np.issubdtype(type(target), int)
