"""Pathology FMs from kaiko.ai."""

from typing import Tuple

from torch import nn

from eva.core.models import transforms, wrappers
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/kaiko_vits16")
def kaiko_vits16(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    concat_mean_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the ViTS-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    return wrappers.TorchHubModel(
        model_name="vits16",
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model_kwargs={"dynamic_img_size": dynamic_img_size, "out_indices": out_indices},
        forward_features=concat_mean_patch_tokens,
        tensor_transforms=(
            transforms.ExtractCLSFeatures(concat_mean_patch_tokens=concat_mean_patch_tokens)
            if concat_mean_patch_tokens
            else None
        ),
    )


@register_model("pathology/kaiko_vits8")
def kaiko_vits8(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    concat_mean_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the ViTS-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    return wrappers.TorchHubModel(
        model_name="vits8",
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model_kwargs={"dynamic_img_size": dynamic_img_size, "out_indices": out_indices},
        forward_features=concat_mean_patch_tokens,
        tensor_transforms=(
            transforms.ExtractCLSFeatures(concat_mean_patch_tokens=concat_mean_patch_tokens)
            if concat_mean_patch_tokens
            else None
        ),
    )


@register_model("pathology/kaiko_vitb16")
def kaiko_vitb16(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    concat_mean_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the ViTB-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    return wrappers.TorchHubModel(
        model_name="vitb16",
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model_kwargs={"dynamic_img_size": dynamic_img_size, "out_indices": out_indices},
        forward_features=concat_mean_patch_tokens,
        tensor_transforms=(
            transforms.ExtractCLSFeatures(concat_mean_patch_tokens=concat_mean_patch_tokens)
            if concat_mean_patch_tokens
            else None
        ),
    )


@register_model("pathology/kaiko_vitb8")
def kaiko_vitb8(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    concat_mean_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the ViTB-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    return wrappers.TorchHubModel(
        model_name="vitb8",
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model_kwargs={"dynamic_img_size": dynamic_img_size, "out_indices": out_indices},
        forward_features=concat_mean_patch_tokens,
        tensor_transforms=(
            transforms.ExtractCLSFeatures(concat_mean_patch_tokens=concat_mean_patch_tokens)
            if concat_mean_patch_tokens
            else None
        ),
    )


@register_model("pathology/kaiko_vitl14")
def kaiko_vitl14(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    concat_mean_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the ViTL-14 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        concat_mean_patch_tokens: Concat the CLS token with mean aggregated patch tokens.

    Returns:
        The model instance.
    """
    return wrappers.TorchHubModel(
        model_name="vitl14",
        repo_or_dir="kaiko-ai/towards_large_pathology_fms",
        model_kwargs={"dynamic_img_size": dynamic_img_size, "out_indices": out_indices},
        forward_features=concat_mean_patch_tokens,
        tensor_transforms=(
            transforms.ExtractCLSFeatures(concat_mean_patch_tokens=concat_mean_patch_tokens)
            if concat_mean_patch_tokens
            else None
        ),
    )
