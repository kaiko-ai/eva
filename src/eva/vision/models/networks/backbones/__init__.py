"""Vision Model Backbones API."""

from eva.vision.models.networks.backbones import pathology, universal
from eva.vision.models.networks.backbones.registry import BackboneModelRegistry, register_model

__all__ = ["pathology", "universal", "BackboneModelRegistry", "register_model"]
