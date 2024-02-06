"""Normalization data structure."""

import dataclasses
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class Normalize:
    """Holds the normalization values for an image tensor."""

    mean: Sequence[float]
    """Sequence of means for each channel."""

    std: Sequence[float]
    """Sequence of standard deviations for each channel."""
