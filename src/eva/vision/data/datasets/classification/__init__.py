"""Image classification datasets API."""

from eva.vision.data.datasets.classification.bach import BACH
from eva.vision.data.datasets.classification.crc_he import CRC_HE
from eva.vision.data.datasets.classification.crc_he_nonorm import CRC_HE_NONORM
from eva.vision.data.datasets.classification.mhist import MHIST
from eva.vision.data.datasets.classification.patch_camelyon import PatchCamelyon
from eva.vision.data.datasets.classification.total_segmentator import TotalSegmentatorClassification

__all__ = [
    "BACH",
    "CRC_HE",
    "CRC_HE_NONORM",
    "MHIST",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
]
