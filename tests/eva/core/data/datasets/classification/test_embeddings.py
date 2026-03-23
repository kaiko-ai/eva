"""Tests for the embeddings datasets."""

import os
from typing import Literal

import pytest
import torch

from eva.core.data.datasets import classification


@pytest.mark.parametrize("split", ["train", "val"])
def test_embedding_dataset(embeddings_dataset: classification.EmbeddingsClassificationDataset):
    """Tests that the dataset returns data in the expected format."""
    # assert data sample is a tuple
    sample = embeddings_dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    embeddings, target = sample
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (8,)
    assert isinstance(target, torch.Tensor)
    assert target in [0, 1]


@pytest.fixture(scope="function")
def embeddings_dataset(
    split: Literal["train", "val", "test"], root_dir: str
) -> classification.EmbeddingsClassificationDataset:
    """EmbeddingsClassificationDataset dataset fixture."""
    dataset = classification.EmbeddingsClassificationDataset(
        root=root_dir,
        manifest_file="manifest.csv",
        split=split,
    )
    dataset.prepare_data()
    dataset.setup()
    return dataset


@pytest.fixture(scope="function")
def root_dir(assets_path: str):
    """Returns the root directory of the test embeddings dataset."""
    return os.path.join(assets_path, "core", "datasets", "embeddings")
