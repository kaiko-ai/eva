"""Vision Transformers base universal backbones."""

from typing import List

import timm
from torch import nn


def vit_small_patch16_224_random(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes a ViTS-16 baseline model with random weights.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return timm.create_model(
        model_name="vit_small_patch16_224_dino",
        pretrained=False,
        features_only=out_indices is not None,
        out_indices=out_indices,
        dynamic_img_size=dynamic_img_size,
    )


def vit_small_patch16_224_dino(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes a ViTS-16 baseline model pretrained on imagenet.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return timm.create_model(
        model_name="vit_small_patch16_224_dino",
        pretrained=True,
        features_only=out_indices is not None,
        out_indices=out_indices,
        dynamic_img_size=dynamic_img_size,
    )
