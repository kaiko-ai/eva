"""Vision datasets API."""

from eva.vision.data.datasets.bach import Bach
from eva.vision.data.datasets.patch_camelyon import PatchCamelyon
from eva.vision.data.datasets.vision import VisionDataset

__all__ = ["VisionDataset", "Bach", "PatchCamelyon"]
