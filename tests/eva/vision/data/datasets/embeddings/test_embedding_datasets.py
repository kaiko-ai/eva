"""Tests for the embedding datasets."""

import os

import numpy as np
import pytest
import torch

from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset


def test_patch_embedding_dataset(patches_manifest_path: str, root_dir: str):
    """Test that the patch level dataset has the correct length and item shapes/types."""
    ds = PatchEmbeddingDataset(
        manifest_path=patches_manifest_path,
        root=root_dir,
        split="train",
    )
    ds.setup()

    expected_shape = (8,)
    assert len(ds) == 3
    for i in range(len(ds)):
        assert isinstance(ds[i], tuple)
        assert len(ds[i]) == 2
        embedding, target = ds[i]
        assert embedding.shape == expected_shape
        assert np.issubdtype(type(target), int)


def test_slide_embedding_dataset(slides_manifest_path: str, root_dir: str):
    """Test that the slide level dataset has the correct length and embedding tensor shapes."""
    ds = SlideEmbeddingDataset(
        manifest_path=slides_manifest_path,
        root=root_dir,
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


@pytest.fixture()
def patches_manifest_path(assets_path: str) -> str:
    """Path to a fake patch level manifest."""
    return os.path.join(assets_path, "vision/manifests/embeddings/patches.csv")


@pytest.fixture()
def slides_manifest_path(assets_path: str) -> str:
    """Path to a fake patch level manifest."""
    return os.path.join(assets_path, "vision/manifests/embeddings/slides.csv")


@pytest.fixture()
def root_dir(assets_path: str) -> str:
    """Root directory for the fake embeddings."""
    return os.path.join(assets_path, "vision/datasets/embeddings")
