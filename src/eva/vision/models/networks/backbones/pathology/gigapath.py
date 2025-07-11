"""Pathology FMs from other/mixed entities."""

from typing import Tuple

import timm
from torch import nn

from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("pathology/prov_gigapath")
def prov_gigapath(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """Initializes the Prov-GigaPath pathology FM.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return timm.create_model(
        model_name="hf_hub:prov-gigapath/prov-gigapath",
        pretrained=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )
