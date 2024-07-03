"""Segmentation datasets API."""

from eva.vision.data.datasets.segmentation.base import ImageSegmentation
<<<<<<< HEAD
from eva.vision.data.datasets.segmentation.embeddings import EmbeddingsSegmentationDataset
from eva.vision.data.datasets.segmentation.total_segmentator import TotalSegmentator2D

__all__ = ["ImageSegmentation", "EmbeddingsSegmentationDataset", "TotalSegmentator2D"]
=======
from eva.vision.data.datasets.segmentation.consep import CoNSeP
from eva.vision.data.datasets.segmentation.monusac import MoNuSAC
from eva.vision.data.datasets.segmentation.total_segmentator_2d import TotalSegmentator2D

__all__ = ["ImageSegmentation", "CoNSeP", "MoNuSAC", "TotalSegmentator2D"]
>>>>>>> main
