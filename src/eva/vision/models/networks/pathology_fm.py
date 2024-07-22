"""Helper wrapper class for Pathology FMs."""

from typing import Any

import torch
from torch import nn


def load_pathology_fm(model_name: str, **kwargs: Any) -> nn.Module:
    """."""
    pathology_fm_lib = {
        "kaiko_vits16": torch.hub.load(
            repo_or_dir="kaiko-ai/towards_large_pathology_fms",
            model="vits16",
            trust_repo=True,
            **kwargs,
        ),
        "kaiko_vits8": torch.hub.load(
            repo_or_dir="kaiko-ai/towards_large_pathology_fms",
            model="vits8",
            trust_repo=True,
            **kwargs,
        ),
        "kaiko_vitb16": torch.hub.load(
            repo_or_dir="kaiko-ai/towards_large_pathology_fms",
            model="vitb16",
            trust_repo=True,
            **kwargs,
        ),
        "kaiko_vitb8": torch.hub.load(
            repo_or_dir="kaiko-ai/towards_large_pathology_fms",
            model="vitb8",
            trust_repo=True,
            **kwargs,
        ),
    }
    model = pathology_fm_lib.get(model_name)
    if model is None:
        raise ValueError("")

    return model


class PathologyFM(nn.Module):
    """Pathology FM model wrapper."""

    def __init__(self, model: str) -> None:
        pass
