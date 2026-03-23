"""Embedding cllassification datasets API."""

from eva.core.data.datasets.classification.embeddings import EmbeddingsClassificationDataset
from eva.core.data.datasets.classification.multi_embeddings import (
    MultiEmbeddingsClassificationDataset,
)

__all__ = ["EmbeddingsClassificationDataset", "MultiEmbeddingsClassificationDataset"]
