"""Embedding callback writers."""

from eva.core.callbacks.writers.embeddings.classification import ClassificationEmbeddingsWriter
from eva.core.callbacks.writers.embeddings.segmentation import SegmentationEmbeddingsWriter

__all__ = ["ClassificationEmbeddingsWriter", "SegmentationEmbeddingsWriter"]
