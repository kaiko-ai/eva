"""Datasets API."""

from eva.core.data.datasets.base import Dataset, MapDataset
from eva.core.data.datasets.classification import (
    EmbeddingsClassificationDataset,
    MultiEmbeddingsClassificationDataset,
)
from eva.core.data.datasets.dataset import TorchDataset
from eva.core.data.datasets.typings import DataSample

__all__ = [
    "Dataset",
    "MapDataset",
    "EmbeddingsClassificationDataset",
    "MultiEmbeddingsClassificationDataset",
    "TorchDataset",
    "DataSample",
]
