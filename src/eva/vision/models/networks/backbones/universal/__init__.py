"""Universal Vision Model Backbones API."""

from eva.vision.models.networks.backbones.universal.vit import (
    vit_base_patch16_224_dino_1chan,
    vit_small_patch16_224_dino,
    vit_small_patch16_224_dino_1chan,
    vit_small_patch16_224_random,
)

__all__ = [
    "vit_small_patch16_224_dino",
    "vit_small_patch16_224_random",
    "vit_small_patch16_224_dino_1chan",
    "vit_base_patch16_224_dino_1chan",
]
