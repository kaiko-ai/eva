"""timm backbones."""

import functools
from typing import Tuple

import timm
from loguru import logger
from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import backbone_registry


def timm_model(
    model_name: str,
    checkpoint_path: str | None = None,
    pretrained: bool = False,
    dynamic_img_size: bool = True,
    out_indices: int | Tuple[int, ...] | None = None,
    **kwargs,
) -> nn.Module:
    """Initializes any ViT model from timm with weights from a specified checkpoint.

    Args:
        model_name: The name of the model to load.
        checkpoint_path: The path to the checkpoint file.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.
        **kwargs: Additional arguments to pass to the model

    Returns:
        The VIT model instance.
    """
    logger.info(
        f"Loading timm model {model_name}"
        + (f"using checkpoint {checkpoint_path}" if checkpoint_path else "")
    )
    return wrappers.TimmModel(
        model_name=model_name,
        checkpoint_path=checkpoint_path or "",
        pretrained=pretrained,
        out_indices=out_indices,
        model_kwargs=kwargs,
    )


backbone_registry._registry.update(
    {
        f"timm/{model_name}": functools.partial(timm_model, model_name=model_name)
        for model_name in timm.list_models()
    }
)
