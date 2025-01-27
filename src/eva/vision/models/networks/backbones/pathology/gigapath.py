"""Pathology FMs from other/mixed entities."""

from typing import Tuple

from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/prov_gigapath")
def prov_gigapath(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    concat_mean_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the Prov-GigaPath pathology FM.

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
        model_name="hf_hub:prov-gigapath/prov-gigapath",
        pretrained=True,
        out_indices=out_indices,
        model_kwargs={
            "dynamic_img_size": dynamic_img_size,
        },
        concat_mean_patch_tokens=concat_mean_patch_tokens,
    )
