"""Image classification datasets API."""

from eva.vision.data.datasets.classification.bach import Bach
from eva.vision.data.datasets.classification.patch_camelyon import PatchCamelyon

__all__ = ["Bach", "PatchCamelyon"]
