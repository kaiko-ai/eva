"""Universal Vision Model Backbones API."""

from eva.vision.models.networks.backbones.universal.timm import timm_model
from eva.vision.models.networks.backbones.universal.vit import (
    vit_small_patch16_224_dino,
    vit_small_patch16_224_random,
)

__all__ = ["vit_small_patch16_224_dino", "vit_small_patch16_224_random", "timm_model"]
