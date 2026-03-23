"""Pathology FMs from Bioptimus."""

from typing import Tuple

import timm
from torch import nn

from eva.core.models import transforms
from eva.vision.models import wrappers
from eva.vision.models.networks.backbones import _utils
from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("pathology/bioptimus_h_optimus_0")
def bioptimus_h_optimus_0(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """Initializes the H-Optimus-0 pathology FM by Bioptimus.

    See https://huggingface.co/bioptimus/H-optimus-0 for details.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return timm.create_model(
        model_name="hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )


@backbone_registry.register("pathology/bioptimus_h0_mini")
def bioptimus_h0_mini(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    hf_token: str | None = None,
    include_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes H0-mini (ViT-B) pathology FM by Bioptimus.

    This model was distilled from H-Optimus-0 on 40M TCGA tiles.

    See https://huggingface.co/bioptimus/H0-mini for details.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        hf_token: HuggingFace token to download the model.
        include_patch_tokens: Whether to combine the mean aggregated patch tokens with cls token.

    Returns:
        The model instance.
    """
    _utils.huggingface_login(hf_token)
    return wrappers.TimmModel(
        model_name="hf-hub:bioptimus/H0-mini",
        out_indices=out_indices,
        pretrained=True,
        model_kwargs={
            "dynamic_img_size": dynamic_img_size,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": nn.SiLU,
        },
        transforms=(
            transforms.ExtractCLSFeatures(include_patch_tokens=include_patch_tokens)
            if out_indices is None
            else None
        ),
    )
