"""Vision Model Backbones API."""

from eva.vision.models.networks.backbones import pathology, radiology, timm, universal
from eva.vision.models.networks.backbones.registry import BackboneModelRegistry, register_model

__all__ = [
    "radiology",
    "pathology",
    "timm",
    "universal",
    "BackboneModelRegistry",
    "register_model",
]
