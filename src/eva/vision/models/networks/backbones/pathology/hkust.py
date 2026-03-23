"""Pathology FMs from Hong Kong University of Science and Technology."""

import re
from typing import Tuple

import timm
from torch import nn

from eva.core.models.wrappers import _utils
from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("pathology/hkust_gpfm")
def hkust_gpfm(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """Initializes GPFM model from Hong Kong University of Science and Technology.

    Ma, J., Guo, Z., Zhou, F., Wang, Y., Xu, Y., et al. (2024).
    Towards a generalizable pathology foundation model via unified knowledge
    distillation (arXiv No. 2407.18449). arXiv. https://arxiv.org/abs/2407.18449

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return timm.create_model(
        model_name="vit_large_patch14_dinov2",
        pretrained=True,
        pretrained_cfg={
            "state_dict": _load_state_dict(),
            "num_classes": 0,
        },
        out_indices=out_indices,
        features_only=out_indices is not None,
        **{
            "img_size": 224,
            "patch_size": 14,
            "init_values": 1e-5,
            "qkv_bias": True,
            "dynamic_img_size": dynamic_img_size,
        },
    )


def _load_state_dict() -> dict:
    """Loads the state dict with model weights from github."""
    state_dict = _utils.load_state_dict_from_url(
        url="https://github.com/birkhoffkiki/GPFM/releases/download/ckpt/GPFM.pth",
        md5="0dc7e345de84f385d09c8c782b4b3236",
    )
    return _convert_state_dict(state_dict["teacher"])


def _convert_state_dict(state_dict: dict) -> dict:
    """Rename state dict keys to match timm's format."""
    state_dict = {
        re.sub(r"blocks\.\d+\.(\d+)", r"blocks.\1", key.replace("backbone.", "")): value
        for key, value in state_dict.items()
    }
    remove_keys = ["mask_token"] + [key for key in state_dict.keys() if "dino_head" in key]
    for key in remove_keys:
        state_dict.pop(key)
    return state_dict
