"""Schema definitions for dataset classes."""

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class TransformsSchema:
    """Schema for dataset transforms."""

    text: Callable | None = None
    """Text transformation"""

    target: Callable | None = None
    """Target transformation"""

    prediction: Callable | None = None
    """Prediction transformation"""
