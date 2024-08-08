"""Pathology FMs from owkin."""

from typing import Tuple

from torch import nn

from eva import models
from eva.core.models import transforms
from eva.vision.models.networks.backbones.registry import register_model


@register_model("pathology/owkin_phikon")
def owkin_phikon(out_indices: int | Tuple[int, ...] | None = None) -> nn.Module:
    """Initializes the phikon pathology FM by owkin (https://huggingface.co/owkin/phikon).

    Args:
        out_indices: Weather and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.

    Returns:
        The model instance.
    """
    if out_indices is None:
        tensor_transforms = transforms.ExtractCLSFeatures()
    elif out_indices == 1:
        tensor_transforms = transforms.ExtractPatchFeatures()
    else:
        raise ValueError(f"out_indices={out_indices} is not supported for phikon")

    return models.HuggingFaceModel(
        model_name_or_path="owkin/phikon", tensor_transforms=tensor_transforms
    )
