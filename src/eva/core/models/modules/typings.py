"""Type annotations for model modules."""

from typing import Any, Dict, List, NamedTuple

import lightning.pytorch as pl
import torch
from torch import nn

MODEL_TYPE = nn.Module | pl.LightningModule
"""The expected model type."""


class INPUT_BATCH(NamedTuple):
    """The default input batch data scheme."""

    data: torch.Tensor | List[torch.Tensor]
    """The data batch."""

    targets: torch.Tensor | None = None
    """The target batch."""

    metadata: Dict[str, Any] | None = None
    """The associated metadata."""


class INPUT_TENSOR_BATCH(NamedTuple):
    """Tensor based input batch data scheme."""

    data: torch.Tensor
    """The data batch."""

    targets: torch.Tensor
    """The target batch."""

    metadata: Dict[str, Any] | None = None
    """The associated metadata."""
