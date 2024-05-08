"""Segmentation datasets API."""

from eva.vision.data.datasets.segmentation.base import ImageSegmentation
from eva.vision.data.datasets.segmentation.bcss import BCSS
from eva.vision.data.datasets.segmentation.total_segmentator import TotalSegmentator2D

__all__ = ["ImageSegmentation", "BCSS", "TotalSegmentator2D"]
