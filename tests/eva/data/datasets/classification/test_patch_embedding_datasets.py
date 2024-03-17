"""Tests for the patch embedding datasets."""

import os
from typing import Tuple

import numpy as np
import pytest
import torch

from eva.data.datasets import classification


@pytest.mark.parametrize(
    "split, embeddings_shape",
    [("train", (8,)), ("val", (8,))],
)
def test_patch_embedding_dataset(
    patch_embeddings_dataset: classification.PatchEmbeddingsDataset,
    embeddings_shape: Tuple[int, ...],
):
    """Test that the PatchEmbeddingsDataset level dataset."""
    # assert data sample is a tuple
    sample = patch_embeddings_dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    embeddings, target = sample
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == embeddings_shape
    assert isinstance(target, np.ndarray)
    assert target in [0, 1]


@pytest.fixture(scope="function")
def patch_embeddings_dataset(split: str, assets_path: str) -> classification.PatchEmbeddingsDataset:
    """PatchEmbeddingsDataset dataset fixture."""
    dataset = classification.PatchEmbeddingsDataset(
        root=os.path.join(assets_path, "core", "datasets", "embeddings", "patch"),
        manifest_file="manifest.csv",
        split=split,
    )
    dataset.prepare_data()
    dataset.setup()
    return dataset
