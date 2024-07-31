"""Baseline FMs."""

from typing import List

import torch
from torch import nn

from eva.vision.models.networks.backbones.pathology._registry import PathologyModelRegistry


@PathologyModelRegistry.register("vits16_random")
def vits16_random(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes a ViTS-16 baseline model with random weights.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return torch.hub.load(
        repo_or_dir="facebookresearch/dino:main",
        model="dino_vits16",
        pretrained=False,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )


@PathologyModelRegistry.register("vits16_imagenet")
def vits16_imagenet(
    dynamic_img_size: bool = True, out_indices: int | List[int] | None = None
) -> nn.Module:
    """Initializes a ViTS-16 baseline model pretrained on imagenet.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return torch.hub.load(
        repo_or_dir="facebookresearch/dino:main",
        model="dino_vits16",
        pretrained=True,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
    )
