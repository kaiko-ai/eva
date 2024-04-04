"""Datasets API."""

from eva.core.data.datasets.embeddings.base import EmbeddingsDataset
from eva.core.data.datasets.embeddings.classification import (
    EmbeddingsClassificationDataset,
    MultiEmbeddingsClassificationDataset,
)

__all__ = [
    "EmbeddingsDataset",
    "EmbeddingsClassificationDataset",
    "MultiEmbeddingsClassificationDataset",
]
