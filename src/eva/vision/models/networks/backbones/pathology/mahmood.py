"""Pathology FMs from MahmoodLab."""

import os
from typing import Tuple

import huggingface_hub
from loguru import logger
from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/mahmood_uni")
def mahmood_uni(
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    hf_token: str | None = None,
    download_dir: str = "~/.cache/eva",
) -> nn.Module:
    """Initializes UNI model from MahmoodLab.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Weather and which multi-level patch embeddings to return.
        hf_token: HuggingFace token to download the model.
        download_dir: Directory to download the model checkpoint.

    Returns:
        The model instance.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Please provide a HuggingFace token to download the model. "
            "You can either pass it as an argument or set the env variable HF_TOKEN."
        )

    checkpoint_path = os.path.join(download_dir, "pytorch_model.bin")

    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading the model checkpoint to {download_dir} ...")
        os.makedirs(download_dir, exist_ok=True)
        huggingface_hub.login(token=token)
        huggingface_hub.hf_hub_download(
            "MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=download_dir,
            force_download=True,
        )

    return wrappers.TimmModel(
        model_name="vit_large_patch16_224",
        out_indices=out_indices,
        model_kwargs={
            "init_values": 1e-5,
            "dynamic_img_size": dynamic_img_size,
        },
        checkpoint_path=checkpoint_path,
    )