"""Pathology FMs from owkin."""

from typing import Tuple

from torch import nn

from eva.vision.models.networks.backbones import _utils
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/owkin_phikon")
def owkin_phikon(
    out_indices: int | Tuple[int, ...] | None = None, concat_mean_patch_tokens: bool = False
) -> nn.Module:
    """Initializes the phikon pathology FM by owkin (https://huggingface.co/owkin/phikon).

    Args:
        out_indices: Whether and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    transform_args = {"concat_mean_patch_tokens": True} if concat_mean_patch_tokens else None
    return _utils.load_hugingface_model(
        model_name="owkin/phikon", out_indices=out_indices, transform_args=transform_args
    )


@register_model("pathology/owkin_phikon_v2")
def owkin_phikon_v2(
    out_indices: int | Tuple[int, ...] | None = None, concat_mean_patch_tokens: bool = False
) -> nn.Module:
    """Initializes the phikon-v2 pathology FM by owkin (https://huggingface.co/owkin/phikon-v2).

    Args:
        out_indices: Whether and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    transform_args = {"concat_mean_patch_tokens": True} if concat_mean_patch_tokens else None
    return _utils.load_hugingface_model(
        model_name="owkin/phikon-v2", out_indices=out_indices, transform_args=transform_args
    )
