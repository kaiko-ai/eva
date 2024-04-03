"""Datasets API."""

from eva.core.data.datasets.base import Dataset
from eva.core.data.datasets.dataset import TorchDataset
from eva.core.data.datasets.embeddings.classification import EmbeddingsClassificationDataset

__all__ = ["Dataset", "EmbeddingsClassificationDataset", "TorchDataset"]
