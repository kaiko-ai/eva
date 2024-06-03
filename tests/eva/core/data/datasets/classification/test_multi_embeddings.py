"""Tests for the embeddings datasets."""

import os
from typing import Literal, Tuple

import numpy as np
import pytest
import torch
import torch.nn

from eva.core.data import transforms
from eva.core.data.datasets import classification


@pytest.mark.parametrize(
    "split, expected_shape",
    [("train", (9, 8)), ("val", (5, 8))],
)
def test_embedding_dataset(
    embeddings_dataset: classification.MultiEmbeddingsClassificationDataset,
    expected_shape: Tuple[int, ...],
):
    """Tests that the dataset returns data in the expected format."""
    # assert data sample is a tuple
    sample = embeddings_dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    embeddings, target = sample
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == expected_shape
    assert isinstance(target, np.ndarray)
    assert target in [0, 1]


@pytest.mark.parametrize(
    "split, n_samples, pad_size, pad_value",
    [("train", 3, 50, float("-inf")), ("val", 4, 7, -1)],
)
def test_embedding_dataset_with_transform(
    embeddings_dataset_with_transform: classification.MultiEmbeddingsClassificationDataset,
    n_samples: int,
    pad_size: int,
    pad_value: int | float,
):
    """Tests that the dataset returns data in the expected format with transforms."""
    # assert data sample is a tuple
    sample = embeddings_dataset_with_transform[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    # assert the format of the `image` and `target`
    embeddings, target = sample
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (pad_size, 8)
    assert isinstance(target, np.ndarray)
    assert target in [0, 1]
    # assert the number of padded entries
    mask = embeddings == pad_value
    mask = mask.all(dim=-1, keepdim=True)
    assert mask.sum() == pad_size - n_samples


@pytest.fixture(scope="function")
def embeddings_dataset(
    split: Literal["train", "val", "test"], root_dir: str
) -> classification.MultiEmbeddingsClassificationDataset:
    """EmbeddingsClassificationDataset dataset fixture."""
    dataset = classification.MultiEmbeddingsClassificationDataset(
        root=root_dir,
        manifest_file="manifest.csv",
        split=split,
    )
    dataset.prepare_data()
    dataset.setup()
    return dataset


@pytest.fixture(scope="function")
def embeddings_dataset_with_transform(
    split: Literal["train", "val", "test"],
    root_dir: str,
    n_samples: int,
    pad_size: int,
    pad_value: int | float,
) -> classification.MultiEmbeddingsClassificationDataset:
    """EmbeddingsClassificationDataset dataset fixture."""

    def _embeddings_transforms(tensor: torch.Tensor) -> torch.Tensor:
        tensor = transforms.SampleFromAxis(n_samples)(tensor)
        return transforms.Pad2DTensor(pad_size, pad_value)(tensor)

    dataset = classification.MultiEmbeddingsClassificationDataset(
        root=root_dir,
        manifest_file="manifest.csv",
        split=split,
        embeddings_transforms=_embeddings_transforms,
    )
    dataset.prepare_data()
    dataset.setup()
    return dataset


@pytest.fixture(scope="function")
def root_dir(assets_path: str):
    """Returns the root directory of the test embeddings dataset."""
    return os.path.join(assets_path, "core", "datasets", "multi-embeddings")
