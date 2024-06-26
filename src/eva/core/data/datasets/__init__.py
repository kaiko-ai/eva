"""Datasets API."""

from eva.core.data.datasets.base import Dataset
from eva.core.data.datasets.classification import (
    EmbeddingsClassificationDataset,
    MultiEmbeddingsClassificationDataset,
)
from eva.core.data.datasets.dataset import TorchDataset

__all__ = [
    "Dataset",
    "EmbeddingsClassificationDataset",
    "MultiEmbeddingsClassificationDataset",
    "TorchDataset",
]
