"""Vision Model Wrappers API."""

from eva.vision.models.wrappers.from_registry import ModelFromRegistry
from eva.vision.models.wrappers.from_timm import TimmModel

__all__ = ["ModelFromRegistry", "TimmModel"]
