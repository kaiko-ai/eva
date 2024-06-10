"""Segmentation datasets API."""

from eva.vision.data.datasets.segmentation.base import ImageSegmentation
from eva.vision.data.datasets.segmentation.embeddings import EmbeddingsSegmentationDataset
from eva.vision.data.datasets.segmentation.total_segmentator import TotalSegmentator2D

__all__ = ["ImageSegmentation", "EmbeddingsSegmentationDataset", "TotalSegmentator2D"]
