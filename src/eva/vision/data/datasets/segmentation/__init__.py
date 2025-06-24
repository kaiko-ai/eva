"""Segmentation datasets API."""

from eva.vision.data.datasets.segmentation.bcss import BCSS
from eva.vision.data.datasets.segmentation.btcv import BTCV
from eva.vision.data.datasets.segmentation.consep import CoNSeP
from eva.vision.data.datasets.segmentation.embeddings import EmbeddingsSegmentationDataset
from eva.vision.data.datasets.segmentation.lits17 import LiTS17
from eva.vision.data.datasets.segmentation.monusac import MoNuSAC

__all__ = [
    "BCSS",
    "BTCV",
    "CoNSeP",
    "EmbeddingsSegmentationDataset",
    "LiTS17",
    "MoNuSAC",
]
