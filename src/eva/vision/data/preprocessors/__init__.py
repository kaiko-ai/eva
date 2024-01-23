"""Processors API."""
from eva.vision.data.preprocessors.bach import BachPreprocessor
from eva.vision.data.preprocessors.patch_camelyon import PatchCamelyonPreprocessor

__all__ = ["BachPreprocessor", "PatchCamelyonPreprocessor"]
