"""Utilis for backbone networks."""

import os
from typing import Any, Dict, Tuple

import huggingface_hub
from torch import nn

from eva import models
from eva.core.models import transforms


def load_hugingface_model(
    model_name: str,
    out_indices: int | Tuple[int, ...] | None,
    model_kwargs: Dict[str, Any] | None = None,
    transform_args: Dict[str, Any] | None = None,
) -> nn.Module:
    """Helper function to load HuggingFace models.

    Args:
        model_name: The model name to load.
        out_indices: Whether and which multi-level patch embeddings to return.
            Currently only out_indices=1 is supported.
        model_kwargs: The arguments used for instantiating the model.
        transform_args: The arguments used for instantiating the transform.

    Returns: The model instance.
    """
    if out_indices is None:
        tensor_transforms = transforms.ExtractCLSFeatures(**(transform_args or {}))
    elif out_indices == 1:
        tensor_transforms = transforms.ExtractPatchFeatures(**(transform_args or {}))
    else:
        raise ValueError(f"out_indices={out_indices} is currently not supported.")

    return models.HuggingFaceModel(
        model_name_or_path=model_name,
        transforms=tensor_transforms,
        model_kwargs=model_kwargs,
    )


def huggingface_login(hf_token: str | None = None):
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Please provide a HuggingFace token to download the model. "
            "You can either pass it as an argument or set the env variable HF_TOKEN."
        )
    huggingface_hub.login(token=token)
