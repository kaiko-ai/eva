"""Vision Datasets API."""

from eva.vision.data.datasets.classification import (
    BACH,
    CRC,
    MHIST,
    PANDA,
    Camelyon16,
    PatchCamelyon,
    WsiClassificationDataset,
)
from eva.vision.data.datasets.segmentation import Consep, ImageSegmentation, TotalSegmentator2D
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.datasets.wsi import MultiWsiDataset, WsiDataset

__all__ = [
    "BACH",
    "CRC",
    "MHIST",
    "ImageSegmentation",
    "PatchCamelyon",
    "Consep",
    "PANDA",
    "Camelyon16",
    "TotalSegmentator2D",
    "VisionDataset",
    "WsiDataset",
    "MultiWsiDataset",
    "WsiClassificationDataset",
]
