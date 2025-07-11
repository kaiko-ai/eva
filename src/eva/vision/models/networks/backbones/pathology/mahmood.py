"""Pathology FMs from MahmoodLab."""

from typing import Tuple

import timm
import torch
from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones import _utils
from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("pathology/mahmood_uni")
def mahmood_uni(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    hf_token: str | None = None,
) -> nn.Module:
    """Initializes UNI model from MahmoodLab.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        hf_token: HuggingFace token to download the model.

    Returns:
        The model instance.
    """
    _utils.huggingface_login(hf_token)

    return wrappers.TimmModel(
        model_name="hf-hub:MahmoodLab/uni",
        pretrained=True,
        out_indices=out_indices,
        model_kwargs={
            "init_values": 1e-5,
            "dynamic_img_size": dynamic_img_size,
        },
    )


@backbone_registry.register("pathology/mahmood_uni2_h")
def mahmood_uni2_h(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    hf_token: str | None = None,
) -> nn.Module:
    """Initializes UNI model from MahmoodLab.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        hf_token: HuggingFace token to download the model.

    Returns:
        The model instance.
    """
    _utils.huggingface_login(hf_token)

    return wrappers.TimmModel(
        model_name="hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        out_indices=out_indices,
        model_kwargs={
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": dynamic_img_size,
        },
    )
