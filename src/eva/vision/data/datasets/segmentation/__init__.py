"""Segmentation datasets API."""

from eva.vision.data.datasets.segmentation.base import ImageSegmentation
from eva.vision.data.datasets.segmentation.consep import CoNSeP
from eva.vision.data.datasets.segmentation.total_segmentator_2d import TotalSegmentator2D

__all__ = ["CoNSeP", "ImageSegmentation", "TotalSegmentator2D"]
