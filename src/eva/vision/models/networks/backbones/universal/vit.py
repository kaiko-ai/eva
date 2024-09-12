"""Vision Transformers base universal backbones."""

from typing import Tuple

import timm
from torch import nn

from eva.vision.models.networks.backbones.registry import register_model


@register_model("universal/vit_small_patch16_224_random")
def vit_small_patch16_224_random(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes a ViTS-16 baseline model with random weights.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return timm.create_model(
        model_name="vit_small_patch16_224.dino",
        pretrained=False,
        features_only=out_indices is not None,
        out_indices=out_indices,
        dynamic_img_size=dynamic_img_size,
    )


@register_model("universal/vit_small_patch16_224_dino")
def vit_small_patch16_224_dino(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes a ViTS-16 baseline model pretrained w/ DINO.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return timm.create_model(
        model_name="vit_small_patch16_224.dino",
        pretrained=True,
        features_only=out_indices is not None,
        out_indices=out_indices,
        dynamic_img_size=dynamic_img_size,
    )
