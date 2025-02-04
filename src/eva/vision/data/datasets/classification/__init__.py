"""Image classification datasets API."""

from eva.vision.data.datasets.classification.bach import BACH
from eva.vision.data.datasets.classification.bracs import BRACS
from eva.vision.data.datasets.classification.breakhis import BreaKHis
from eva.vision.data.datasets.classification.camelyon16 import Camelyon16
from eva.vision.data.datasets.classification.crc import CRC
from eva.vision.data.datasets.classification.gleason_arvaniti import GleasonArvaniti
from eva.vision.data.datasets.classification.mhist import MHIST
from eva.vision.data.datasets.classification.panda import PANDA, PANDASmall
from eva.vision.data.datasets.classification.patch_camelyon import PatchCamelyon
from eva.vision.data.datasets.classification.unitopatho import UniToPatho
from eva.vision.data.datasets.classification.wsi import WsiClassificationDataset

__all__ = [
    "BACH",
    "BreaKHis",
    "BRACS",
    "Camelyon16",
    "CRC",
    "GleasonArvaniti",
    "MHIST",
    "PatchCamelyon",
    "UniToPatho",
    "WsiClassificationDataset",
    "PANDA",
    "PANDASmall",
    "Camelyon16",
]
