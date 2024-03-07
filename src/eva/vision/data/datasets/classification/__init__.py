"""Image classification datasets API."""

from eva.vision.data.datasets.classification.bach import BACH
from eva.vision.data.datasets.classification.crc import CRC
from eva.vision.data.datasets.classification.mhist import MHIST
from eva.vision.data.datasets.classification.patch_camelyon import PatchCamelyon
from eva.vision.data.datasets.classification.total_segmentator import TotalSegmentatorClassification

__all__ = [
    "BACH",
    "CRC",
    "MHIST",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
]
