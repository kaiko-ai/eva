"""Embedding cllassification datasets API."""

from eva.core.data.datasets.embeddings.classification.embeddings import (
    EmbeddingsClassificationDataset,
)
from eva.core.data.datasets.embeddings.classification.multi_embeddings import (
    MultiEmbeddingsClassificationDataset,
)

__all__ = ["EmbeddingsClassificationDataset", "MultiEmbeddingsClassificationDataset"]
