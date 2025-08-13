"""Multimodal datasets API."""

from eva.multimodal.data.datasets.patch_camelyon import PatchCamelyonVQA
from eva.multimodal.data.datasets.text_image import TextImageDataset

__all__ = ["TextImageDataset", "PatchCamelyonVQA"]
