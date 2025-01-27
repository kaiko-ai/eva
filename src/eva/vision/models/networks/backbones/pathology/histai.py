"""Pathology FMs from owkin."""

from typing import Tuple

from torch import nn

from eva.vision.models.networks.backbones import _utils
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/histai_hibou_b")
def histai_hibou_b(
    out_indices: int | Tuple[int, ...] | None = None, concat_mean_patch_tokens: bool = False
) -> nn.Module:
    """Initializes the hibou-B pathology FM by hist.ai (https://huggingface.co/histai/hibou-B).

    Uses a customized implementation of the DINOv2 architecture from the transformers
    library to add support for registers, which requires the trust_remote_code=True flag.

    Args:
        out_indices: Whether and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    transform_args = {"num_register_tokens": 4}
    return _utils.load_hugingface_model(
        model_name="histai/hibou-B",
        out_indices=out_indices,
        model_kwargs={"trust_remote_code": True},
        transform_args=(
            transform_args | {"concat_mean_patch_tokens": concat_mean_patch_tokens}
            if concat_mean_patch_tokens
            else transform_args
        ),
    )


@register_model("pathology/histai_hibou_l")
def histai_hibou_l(
    out_indices: int | Tuple[int, ...] | None = None, concat_mean_patch_tokens: bool = False
) -> nn.Module:
    """Initializes the hibou-L pathology FM by hist.ai (https://huggingface.co/histai/hibou-L).

    Uses a customized implementation of the DINOv2 architecture from the transformers
    library to add support for registers, which requires the trust_remote_code=True flag.

    Args:
        out_indices: Whether and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    transform_args = {"num_register_tokens": 4}
    return _utils.load_hugingface_model(
        model_name="histai/hibou-L",
        out_indices=out_indices,
        model_kwargs={"trust_remote_code": True},
        transform_args=(
            transform_args | {"concat_mean_patch_tokens": concat_mean_patch_tokens}
            if concat_mean_patch_tokens
            else transform_args
        ),
    )
