"""Vision Datasets API."""

from eva.vision.data.datasets.bach import Bach
from eva.vision.data.datasets.embeddings import PatchEmbeddingDataset, SlideEmbeddingDataset
from eva.vision.data.datasets.patch_camelyon import PatchCamelyon
from eva.vision.data.datasets.vision import VisionDataset

__all__ = [
    "Bach",
    "PatchEmbeddingDataset",
    "SlideEmbeddingDataset",
    "PatchCamelyon",
    "VisionDataset",
]
