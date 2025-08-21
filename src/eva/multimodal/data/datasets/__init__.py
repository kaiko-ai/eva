"""Multimodal datasets API."""

from eva.multimodal.data.datasets.multiple_choice.patch_camelyon import PatchCamelyon
from eva.multimodal.data.datasets.text_image import TextImageDataset

__all__ = ["TextImageDataset", "PatchCamelyon"]
