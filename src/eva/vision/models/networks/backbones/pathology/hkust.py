"""Pathology FMs from Hong Kong University of Science and Technology."""

import os
import re
from pathlib import Path
from typing import Tuple

import torch
from loguru import logger
from torch import nn
from torchvision.datasets import utils

from eva.vision.data.datasets import structs
from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model


def _download_gpfm() -> Path:
    """Download GPFM model to $HOME/.cache and return file location."""
    filename = "gpfm_github.pth"
    resource = structs.DownloadResource(
        filename=filename,
        url="https://github.com/birkhoffkiki/GPFM/releases/download/ckpt/GPFM.pth",
        md5="0dc7e345de84f385d09c8c782b4b3236",
    )
    root = Path(os.environ["HOME"]) / ".cache"
    utils.download_url(resource.url, root, resource.filename, resource.md5)
    return root / filename


def _convert_gpfm(checkpoint_path: Path) -> Path:
    """Convert GPFM raw checkpoint into timm compatible checkpoint and save it to disk."""
    new_checkpoint_path = Path(checkpoint_path).parent / "gpfm.pth"
    if new_checkpoint_path.exists():
        logger.info("Checkpoint already converted to timm.")
        return new_checkpoint_path
    logger.info("Converting github checkpoint to timm...")
    gpfm_github_state_dict = torch.load(checkpoint_path, weights_only=True)["teacher"]
    gpfm_github_state_dict = {
        re.sub(r"blocks\.\d+\.(\d+)", r"blocks.\1", key.replace("backbone.", "")): value
        for key, value in gpfm_github_state_dict.items()
    }
    remove_keys = ["mask_token"] + [
        key for key in gpfm_github_state_dict.keys() if "dino_head" in key
    ]
    for key in remove_keys:
        gpfm_github_state_dict.pop(key)
    torch.save(gpfm_github_state_dict, new_checkpoint_path)
    logger.info(f"Converted checkpoint saved at {str(new_checkpoint_path)}.")
    return new_checkpoint_path


def process_gpfm() -> str:
    """Download and convert GPFM checkpoint from Github."""
    # Step 1: download from Github
    gpfm_github_checkpoint_path = _download_gpfm()
    # Step 2: convert the checkpoint to make it compatible with timm
    gpfm_timm_checkpoint_path = _convert_gpfm(gpfm_github_checkpoint_path)
    return str(gpfm_timm_checkpoint_path)


@register_model("pathology/hkust_gpfm")
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
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "init_values": 1e-5,
        "qkv_bias": True,
        "dynamic_img_size": dynamic_img_size,
    }
    return wrappers.TimmModel(
        model_name="vit_large_patch14_dinov2",
        pretrained=False,
        checkpoint_path=process_gpfm(),
        out_indices=out_indices,
        model_kwargs=timm_kwargs,
    )
