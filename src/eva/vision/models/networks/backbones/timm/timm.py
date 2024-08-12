"""timm backbones."""

import os
from typing import Tuple

from loguru import logger
from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model


@register_model("timm/timm_model")
def timm_model(
    model_name: str | None = None,
    checkpoint_path: str | None = None,
    pretrained: bool = False,
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
) -> nn.Module:
    """Initializes any ViT model from timm with weights from a specified checkpoint.

    This class is mainly provided to facilitate experimentation. E.g. to load
    a model from a local checkpoint file, without having to manually update
    the default configuration files.

    Args:
        model_name: The name of the model to load. If not specified, will
            load from the environment variable `TIMM_MODEL_NAME`.
        checkpoint_path: The path to the checkpoint file. If not specified,
            will load from the environment variable `CHECKPOINT_PATH`.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The VIT model instance.
    """
    model_name = model_name or os.getenv("TIMM_MODEL_NAME")
    checkpoint_path = checkpoint_path or os.getenv("CHECKPOINT_PATH")
    if not model_name:
        raise ValueError("No model_name is set.")
    if checkpoint_path and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    logger.info(f"Loading timm model {model_name} from checkpoint {checkpoint_path}")
    return wrappers.TimmModel(
        model_name=model_name,
        checkpoint_path=checkpoint_path or "",
        pretrained=pretrained,
        out_indices=out_indices,
        model_kwargs={
            "dynamic_img_size": dynamic_img_size,
        },
    )