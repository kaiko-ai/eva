"""Pathology FMs from Hong Kong University of Science and Technology."""

from typing import Tuple

from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/hkust_gpfm")
def hkust_gpfm(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """Initializes GPFM model from Hong Kong University of Science and Technology.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return wrappers.TimmModel(
        model_name="vit_large_patch14_dinov2",
        pretrained=False,
        checkpoint_path="/path/to/gpfm_checkpoint/GPFM.pt",
        out_indices=out_indices,
        model_kwargs={
            "img_size": 224,
            "patch_size": 14,
            "init_values": 1e-5,
            "qkv_bias": True,
            "dynamic_img_size": dynamic_img_size,
        },
    )
