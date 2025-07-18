"""Pathology FMs from Lunit.

Source: https://github.com/lunit-io/benchmark-ssl-pathology/releases

For training the vit-s models the following standardization parameters were used:

mean: [ 0.70322989, 0.53606487, 0.66096631 ]
std: [ 0.21716536, 0.26081574, 0.20723464 ]
"""

from typing import Tuple

from torch import nn

from eva.vision.models import wrappers
from eva.vision.models.networks.backbones.registry import backbone_registry

VITS_URL_PREFIX = (
    "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
)


@backbone_registry.register("pathology/lunit_vits16")
def lunit_vits16(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTS-16 pathology FM by lunit.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return wrappers.TimmModel(
        model_name="vit_small_patch16_224.dino",
        out_indices=out_indices,
        model_kwargs={
            "dynamic_img_size": dynamic_img_size,
        },
        checkpoint_path=f"{VITS_URL_PREFIX}/dino_vit_small_patch16_ep200.torch",
    )


@backbone_registry.register("pathology/lunit_vits8")
def lunit_vits8(
    dynamic_img_size: bool = True, out_indices: int | Tuple[int, ...] | None = None
) -> nn.Module:
    """Initializes the ViTS-8 pathology FM by lunit.

    Args:
        dynamic_img_size: Support different input image sizes by allowing to change
            the grid size (interpolate abs and/or ROPE pos) in the forward pass.
        out_indices: Whether and which multi-level patch embeddings to return.

    Returns:
        The model instance.
    """
    return wrappers.TimmModel(
        model_name="vit_small_patch8_224.dino",
        out_indices=out_indices,
        model_kwargs={
            "dynamic_img_size": dynamic_img_size,
        },
        checkpoint_path=f"{VITS_URL_PREFIX}/dino_vit_small_patch8_ep200.torch",
    )
