"""torch.hub backbones."""

import functools
from typing import Tuple

import torch
from loguru import logger
from torch import nn

from eva.core.models import wrappers
from eva.vision.models.networks.backbones.registry import BackboneModelRegistry

HUB_REPOS = ["facebookresearch/dinov2:main", "kaiko-ai/towards_large_pathology_fms"]
"""List of torch.hub repositories for which to add the models to the registry."""


def torch_hub_model(
    model_name: str,
    repo_or_dir: str,
    checkpoint_path: str | None = None,
    pretrained: bool = False,
    out_indices: int | Tuple[int, ...] | None = None,
    **kwargs,
) -> nn.Module:
    """Initializes any ViT model from torch.hub with weights from a specified checkpoint.

    Args:
        model_name: The name of the model to load.
        repo_or_dir: The torch.hub repository or local directory to load the model from.
        checkpoint_path: The path to the checkpoint file.
        pretrained: If set to `True`, load pretrained model weights if available.
        out_indices: Whether and which multi-level patch embeddings to return.
        **kwargs: Additional arguments to pass to the model

    Returns:
        The VIT model instance.
    """
    logger.info(
        f"Loading torch.hub model {model_name} from {repo_or_dir}"
        + (f"using checkpoint {checkpoint_path}" if checkpoint_path else "")
    )

    return wrappers.TorchHubModel(
        model_name=model_name,
        repo_or_dir=repo_or_dir,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path or "",
        out_indices=out_indices,
        model_kwargs=kwargs,
    )


BackboneModelRegistry._registry.update(
    {
        f"torchhub/{repo}:{model_name}": functools.partial(
            torch_hub_model, model_name=model_name, repo_or_dir=repo
        )
        for repo in HUB_REPOS
        for model_name in torch.hub.list(repo, verbose=False)
    }
)
