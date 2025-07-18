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
    BTCV,
    CoNSeP,
    EmbeddingsSegmentationDataset,
    LiTS17,
    MoNuSAC,
    MSDTask7Pancreas,
)
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.datasets.wsi import MultiWsiDataset, WsiDataset

__all__ = [
    "BACH",
    "BTCV",
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
    "LiTS17",
    "MSDTask7Pancreas",
    "MoNuSAC",
    "VisionDataset",
    "MultiWsiDataset",
    "WsiDataset",
]
