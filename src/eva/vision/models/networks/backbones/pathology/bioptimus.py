"""Pathology FMs from Bioptimus."""

from typing import Tuple

from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/bioptimus_h_optimus_0")
def bioptimus_h_optimus_0(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    concat_mean_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the h_optimus_0 pathology FM by Bioptimus.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    return wrappers.TimmModel(
        model_name="hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        out_indices=out_indices,
        model_kwargs={
            "dynamic_img_size": dynamic_img_size,
            "init_values": 1e-5,
        },
        concat_mean_patch_tokens=concat_mean_patch_tokens,
    )
