"""Pathology FMs from kaiko.ai."""

from typing import Tuple

import torch
from torch import nn

from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/kaiko_vits16")
def kaiko_vits16(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTS-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vits16",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vits8")
def kaiko_vits8(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTS-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vits8",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitb16")
def kaiko_vitb16(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTB-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitb16",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitb8")
def kaiko_vitb8(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTB-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitb8",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitl14")
def kaiko_vitl14(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTL-14 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return torch.hub.load(  # type: ignore
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitl14",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )
