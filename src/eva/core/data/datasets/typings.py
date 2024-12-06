"""Typing definitions for the datasets module."""

from typing import Any, Dict, NamedTuple

import torch


class DataSample(NamedTuple):
    """The default input batch data scheme."""

    data: torch.Tensor
    """The data batch."""

    targets: torch.Tensor | None = None
    """The target batch."""

    metadata: Dict[str, Any] | None = None
    """The associated metadata."""
