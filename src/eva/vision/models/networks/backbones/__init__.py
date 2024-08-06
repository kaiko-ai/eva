"""Vision Model Backbones API."""

from eva.vision.models.networks.backbones import pathology, universal
from eva.vision.models.networks.backbones.registry import BackboneModelRegistry

__all__ = ["pathology", "universal", "BackboneModelRegistry"]
