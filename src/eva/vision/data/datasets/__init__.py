"""Vision Datasets API."""

from eva.vision.data.datasets.classification import (
    Bach,
    PatchCamelyon,
    TotalSegmentatorClassification,
)
from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset
from eva.vision.data.datasets.segmentation import TotalSegmentator
from eva.vision.data.datasets.vision import VisionDataset

__all__ = [
    "Bach",
    "PatchEmbeddingDataset",
    "SlideEmbeddingDataset",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
    "TotalSegmentator",
    "VisionDataset",
]
