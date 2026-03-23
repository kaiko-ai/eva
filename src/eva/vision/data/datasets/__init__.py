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
    FLARE22,
    WORD,
    CoNSeP,
    EmbeddingsSegmentationDataset,
    KiTS23,
    LiTS17,
    MoNuSAC,
    MSDTask7Pancreas,
)
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.datasets.wsi import MultiWsiDataset, WsiDataset

__all__ = [
    "BACH",
    "BCSS",
    "BreaKHis",
    "BRACS",
    "BTCV",
    "Camelyon16",
    "CoNSeP",
    "CRC",
    "EmbeddingsSegmentationDataset",
    "FLARE22",
    "GleasonArvaniti",
    "KiTS23",
    "LiTS17",
    "MHIST",
    "MoNuSAC",
    "MSDTask7Pancreas",
    "MultiWsiDataset",
    "PANDA",
    "PANDASmall",
    "PatchCamelyon",
    "UniToPatho",
    "VisionDataset",
    "WORD",
    "WsiClassificationDataset",
    "WsiDataset",
]
