"""Pathology FMs from kaiko.ai."""

from typing import List

import torch
from torch import nn

from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/kaiko_vits16")
def kaiko_vits16(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes the vision transformer ViTS-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return torch.hub.load(
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vits16",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vits8")
def kaiko_vits8(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes the vision transformer ViTS-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-8 based foundation model.
    """
    return torch.hub.load(
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vits8",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitb16")
def kaiko_vitb16(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes the vision transformer ViTB-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTB-16 based foundation model.
    """
    return torch.hub.load(
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitb16",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitb8")
def kaiko_vitb8(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes the vision transformer ViTB-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTB-8 based foundation model.
    """
    return torch.hub.load(
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitb8",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@register_model("pathology/kaiko_vitl14")
def kaiko_vitl14(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes the vision transformer ViTL-14 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTL-14 based foundation model.
    """
    return torch.hub.load(
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model="vitl14",
        trust_repo=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )
