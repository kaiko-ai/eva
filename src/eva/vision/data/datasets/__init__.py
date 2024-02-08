"""Vision Datasets API."""

from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset
from eva.vision.data.datasets.total_segmentator import TotalSegmentatorClassification
from src.eva.vision.data.datasets.classification import Bach, PatchCamelyon

__all__ = [
    "Bach",
    "PatchEmbeddingDataset",
    "SlideEmbeddingDataset",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
]
