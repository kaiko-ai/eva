"""Image classification datasets API."""

from eva.vision.data.datasets.classification.bach import BACH
from eva.vision.data.datasets.classification.crc import CRC
from eva.vision.data.datasets.classification.crc_nonorm import CRC_NONORM
from eva.vision.data.datasets.classification.patch_camelyon import PatchCamelyon
from eva.vision.data.datasets.classification.total_segmentator import TotalSegmentatorClassification

__all__ = [
    "BACH",
    "CRC",
    "CRC_NONORM",
    "PatchCamelyon",
    "TotalSegmentatorClassification",
]
