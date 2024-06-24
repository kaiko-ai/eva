"""Segmentation datasets API."""

from eva.vision.data.datasets.segmentation.base import ImageSegmentation
from eva.vision.data.datasets.segmentation.consep import Consep
from eva.vision.data.datasets.segmentation.total_segmentator_2d import TotalSegmentator2D

__all__ = ["Consep", "ImageSegmentation", "TotalSegmentator2D"]
