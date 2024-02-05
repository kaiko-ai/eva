"""Vision Datasets API."""

from eva.vision.data.datasets.embeddings.classification import EmbeddingClassificationDataset
from eva.vision.data.datasets.embeddings.embedding import EmbeddingDataset
from eva.vision.data.datasets.typings import DatasetType
from eva.vision.data.datasets.bach import Bach
from eva.vision.data.datasets.patch_camelyon import PatchCamelyon
from eva.vision.data.datasets.vision import VisionDataset

__all__ = ["VisionDataset", "EmbeddingDataset", "EmbeddingClassificationDataset", "DatasetType", "Bach", "PatchCamelyon"]