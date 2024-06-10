"""Vision Datasets API."""

from eva.vision.data.datasets.classification import BACH, CRC, MHIST, PatchCamelyon
from eva.vision.data.datasets.segmentation import (
    EmbeddingsSegmentationDataset,
    ImageSegmentation,
    TotalSegmentator2D,
)
from eva.vision.data.datasets.vision import VisionDataset

__all__ = [
    "BACH",
    "CRC",
    "MHIST",
    "ImageSegmentation",
    "EmbeddingsSegmentationDataset",
    "PatchCamelyon",
    "TotalSegmentator2D",
    "VisionDataset",
]
