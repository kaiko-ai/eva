"""Vision Model Backbones API."""

from eva.vision.models.networks.backbones import pathology, timm, torchhub, universal
from eva.vision.models.networks.backbones.registry import BackboneModelRegistry, register_model

__all__ = ["pathology", "timm", "torchhub", "universal", "BackboneModelRegistry", "register_model"]
