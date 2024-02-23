"""Vision Datasets API."""

from eva.vision.data.datasets.classification import (
    BACH,
    CRC_HE,
    PatchCamelyon,
    TotalSegmentatorClassification,
)
from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset
from eva.vision.data.datasets.segmentation import TotalSegmentator2D
from eva.vision.data.datasets.vision import VisionDataset

__all__ = [
    "BACH",
    "CRC_HE",
    "PatchEmbeddingDataset",
    "SlideEmbeddingDataset",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
    "TotalSegmentator2D",
    "VisionDataset",
]
