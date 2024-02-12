"""Vision Datasets API."""

from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset
from src.eva.vision.data.datasets.classification import (
    Bach,
    PatchCamelyon,
    TotalSegmentatorClassification,
)

__all__ = [
    "Bach",
    "PatchEmbeddingDataset",
    "SlideEmbeddingDataset",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
]
