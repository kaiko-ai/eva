"""Tests for the embedding datasets."""

import os

import numpy as np
import pytest
import torch

from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset


@pytest.fixture()
def patch_level_manifest_path(assets_path: str) -> str:
    """Path to a fake patch level manifest."""
    return os.path.join(assets_path, "manifests", "embeddings", "patch_level.csv")


@pytest.fixture()
def slide_level_manifest_path(assets_path: str) -> str:
    """Path to a fake patch level manifest."""
    return os.path.join(assets_path, "manifests", "embeddings", "slide_level.csv")


def test_patch_embedding_dataset(patch_level_manifest_path: str, assets_path: str):
    """Test that the patch level dataset has the correct length and item shapes/types."""
    ds = PatchEmbeddingDataset(
        manifest_path=patch_level_manifest_path,
        root_dir=assets_path,
        split="train",
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


def test_slide_embedding_dataset(slide_level_manifest_path: str, assets_path: str):
    """Test that the slide level dataset has the correct length and embedding tensor shapes."""
    ds = SlideEmbeddingDataset(
        manifest_path=slide_level_manifest_path,
        root_dir=assets_path,
        split="train",
        n_patches_per_slide=10,
    )
    ds.setup()

    expected_shape = (10, 8)
    assert len(ds) == 3
    for i in range(len(ds)):
        assert isinstance(ds[i], tuple)
        assert len(ds[i]) == 3
        embedding, target, metadata = ds[i]
        assert embedding.shape == expected_shape
        assert np.issubdtype(type(target), int)
        assert isinstance(metadata, dict)
        assert "mask" in metadata.keys()
        assert isinstance(metadata["mask"], torch.Tensor)
        assert metadata["mask"].shape == (expected_shape[0], 1)
