"""Universal Vision Model Backbones API."""

from eva.vision.models.networks.backbones.universal.vit import (
    vit_small_patch16_224_imagenet,
    vit_small_patch16_224_random,
    vit_timm,
)

__all__ = ["vit_small_patch16_224_imagenet", "vit_small_patch16_224_random", "vit_timm"]
