"""Type annotations for model modules."""
from typing import Any, Dict, NamedTuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from typing_extensions import NotRequired, TypedDict

MODEL_TYPE = Union[nn.Module, pl.LightningModule]
"""The expected model type."""


class TUPLE_INPUT_BATCH(NamedTuple):
    """The tuple input batch data scheme."""

    data: torch.Tensor
    """The data batch."""

    targets: torch.Tensor | None = None
    """The target batch."""

    metadata: Dict[str, Any] | None = None
    """The associated metadata."""


class DICT_INPUT_BATCH(TypedDict):
    """The dictionary input batch data scheme."""

    data: torch.Tensor
    """The data batch."""

    targets: NotRequired[torch.Tensor]
    """The target batch."""

    metadata: NotRequired[torch.Tensor]
    """The associated metadata."""


INPUT_BATCH = TUPLE_INPUT_BATCH | DICT_INPUT_BATCH
"""Combines the tuple and dict input batch."""
