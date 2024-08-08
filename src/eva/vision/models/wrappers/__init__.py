"""Vision Model Wrappers API."""

from eva.vision.models.wrappers.backbone import VisionBackbone
from eva.vision.models.wrappers.from_timm import TimmModel

__all__ = ["TimmModel", "VisionBackbone"]
