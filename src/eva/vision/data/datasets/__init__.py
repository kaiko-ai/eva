"""Vision Datasets API."""

from eva.vision.data.datasets.classification import BACH, CRC, MHIST, PatchCamelyon
from eva.vision.data.datasets.segmentation import Consep, ImageSegmentation, TotalSegmentator2D
from eva.vision.data.datasets.vision import VisionDataset

__all__ = [
    "BACH",
    "CRC",
    "MHIST",
    "ImageSegmentation",
    "PatchCamelyon",
    "Consep",
    "TotalSegmentator2D",
    "VisionDataset",
]
