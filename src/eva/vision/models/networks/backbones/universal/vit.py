"""Vision Transformers base universal backbones."""

import os
from typing import Tuple

import timm
from loguru import logger
from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import register_model


@register_model("universal/vit_small_patch16_224_random")
def vit_small_patch16_224_random(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
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
    return timm.create_model(
        model_name="vit_small_patch16_224.dino",
        pretrained=False,
        features_only=out_indices is not None,
        out_indices=out_indices,
        dynamic_img_size=dynamic_img_size,
    )


@register_model("universal/vit_small_patch16_224_imagenet")
def vit_small_patch16_224_imagenet(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
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
    return timm.create_model(
        model_name="vit_small_patch16_224.dino",
        pretrained=True,
        features_only=out_indices is not None,
        out_indices=out_indices,
        dynamic_img_size=dynamic_img_size,
    )


@register_model("universal/vit_timm")
def vit_timm(
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
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
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
