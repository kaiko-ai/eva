"""Vision Datasets API."""

from eva.vision.data.datasets.classification import (
    BACH,
    PatchCamelyon,
    TotalSegmentatorClassification,
)
from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset
from eva.vision.data.datasets.vision import VisionDataset

__all__ = [
    "BACH",
    "PatchEmbeddingDataset",
    "SlideEmbeddingDataset",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
    "VisionDataset",
]
