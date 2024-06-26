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
from eva.vision.data.datasets.segmentation import CoNSeP, ImageSegmentation, TotalSegmentator2D
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.datasets.wsi import MultiWsiDataset, WsiDataset

__all__ = [
    "BACH",
    "CRC",
    "MHIST",
    "ImageSegmentation",
    "PatchCamelyon",
    "PANDA",
    "Camelyon16",
    "CoNSeP",
    "TotalSegmentator2D",
    "VisionDataset",
    "WsiDataset",
    "MultiWsiDataset",
    "WsiClassificationDataset",
]
