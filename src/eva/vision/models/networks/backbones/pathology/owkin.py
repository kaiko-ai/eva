"""Pathology FMs from owkin."""

from typing import Tuple

from torch import nn

from eva.vision.models.networks.backbones import _utils
from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("pathology/owkin_phikon")
def owkin_phikon(out_indices: int | Tuple[int, ...] | None = None) -> nn.Module:
    """Initializes the phikon pathology FM by owkin (https://huggingface.co/owkin/phikon).

    Args:
        out_indices: Whether and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.

    Returns:
        The model instance.
    """
    return _utils.load_hugingface_model(model_name="owkin/phikon", out_indices=out_indices)


@backbone_registry.register("pathology/owkin_phikon_v2")
def owkin_phikon_v2(out_indices: int | Tuple[int, ...] | None = None) -> nn.Module:
    """Initializes the phikon-v2 pathology FM by owkin (https://huggingface.co/owkin/phikon-v2).

    Args:
        out_indices: Whether and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.

    Returns:
        The model instance.
    """
    return _utils.load_hugingface_model(model_name="owkin/phikon-v2", out_indices=out_indices)
