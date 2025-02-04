"""Vision Datasets API."""

from eva.vision.data.datasets.classification import (
    BACH,
    BRACS,
    CRC,
    MHIST,
    PANDA,
    BreaKHis,
    Camelyon16,
    GleasonArvaniti,
    PANDASmall,
    PatchCamelyon,
    UniToPatho,
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
    "BreaKHis",
    "BRACS",
    "CRC",
    "GleasonArvaniti",
    "MHIST",
    "PANDA",
    "PANDASmall",
    "Camelyon16",
    "PatchCamelyon",
    "UniToPatho",
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
