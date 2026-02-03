"""Multimodal datasets API."""

from eva.multimodal.data.datasets.free_form import QuiltVQA
from eva.multimodal.data.datasets.multiple_choice import PatchCamelyon, PathMMUAtlas
from eva.multimodal.data.datasets.text_image import TextImageDataset

__all__ = ["TextImageDataset", "PatchCamelyon", "PathMMUAtlas", "QuiltVQA"]
