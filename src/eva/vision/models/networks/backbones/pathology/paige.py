"""Pathology FMs from paige.ai.

Source: https://huggingface.co/paige-ai/
"""

from typing import Tuple

import timm
import torch.nn as nn

from eva.core.models import transforms
from eva.vision.models import wrappers
from eva.vision.models.networks.backbones import _utils
from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("pathology/paige_virchow2")
def paige_virchow2(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    hf_token: str | None = None,
    include_patch_tokens: bool = False,
) -> nn.Module:
    """Initializes the Virchow2 pathology FM by paige.ai.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        include_patch_tokens: Whether to combine the mean aggregated patch tokens with cls token.
        hf_token: HuggingFace token to download the model.

    Returns:
        The model instance.
    """
    _utils.huggingface_login(hf_token)
    return wrappers.TimmModel(
        model_name="hf-hub:paige-ai/Virchow2",
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
