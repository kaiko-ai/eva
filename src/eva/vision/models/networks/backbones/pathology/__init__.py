"""Vision Pathology Model Backbones API."""

from eva.vision.models.networks.backbones.pathology import kaiko as kaiko_models
from eva.vision.models.networks.backbones.pathology._registry import PathologyModelRegistry

__all__ = ["PathologyModelRegistry", "kaiko_models"]
