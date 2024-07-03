"""Vision Datasets API."""

<<<<<<< HEAD
from eva.vision.data.datasets.classification import BACH, CRC, MHIST, PatchCamelyon
from eva.vision.data.datasets.segmentation import (
    EmbeddingsSegmentationDataset,
    ImageSegmentation,
=======
from eva.vision.data.datasets.classification import (
    BACH,
    CRC,
    MHIST,
    PANDA,
    Camelyon16,
    PatchCamelyon,
    WsiClassificationDataset,
)
from eva.vision.data.datasets.segmentation import (
    CoNSeP,
    ImageSegmentation,
    MoNuSAC,
>>>>>>> main
    TotalSegmentator2D,
)
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.datasets.wsi import MultiWsiDataset, WsiDataset

__all__ = [
    "BACH",
    "CRC",
    "MHIST",
<<<<<<< HEAD
    "ImageSegmentation",
    "EmbeddingsSegmentationDataset",
=======
    "PANDA",
    "Camelyon16",
>>>>>>> main
    "PatchCamelyon",
    "WsiClassificationDataset",
    "CoNSeP",
    "ImageSegmentation",
    "MoNuSAC",
    "TotalSegmentator2D",
    "VisionDataset",
    "MultiWsiDataset",
    "WsiDataset",
]
