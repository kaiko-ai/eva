"""Vision Datasets API."""

from eva.vision.data.datasets.classification import (
    BACH,
    CRC,
    MHIST,
    PANDA,
    Camelyon16,
    PANDASmall,
    PatchCamelyon,
    WsiClassificationDataset,
)
from eva.vision.data.datasets.segmentation import (
    BCSS,
    CoNSeP,
    EmbeddingsSegmentationDataset,
    ImageSegmentation,
    LiTS,
    LiTSBalanced,
    MoNuSAC,
    TotalSegmentator2D,
)
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.datasets.wsi import MultiWsiDataset, WsiDataset

__all__ = [
    "BACH",
    "BCSS",
    "CRC",
    "MHIST",
    "PANDA",
    "PANDASmall",
    "Camelyon16",
    "PatchCamelyon",
    "WsiClassificationDataset",
    "CoNSeP",
    "EmbeddingsSegmentationDataset",
    "ImageSegmentation",
    "LiTS",
    "LiTSBalanced",
    "MoNuSAC",
    "TotalSegmentator2D",
    "VisionDataset",
    "MultiWsiDataset",
    "WsiDataset",
]
