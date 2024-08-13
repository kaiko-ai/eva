"""Pathology FMs from Bioptimus."""

from typing import Tuple

import timm
from torch import nn

from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/bioptimus_h_optimus_0")
def bioptimus_h_optimus_0(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """Initializes the h_optimus_0 pathology FM by Bioptimus.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return timm.create_model(
        model_name="hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )
