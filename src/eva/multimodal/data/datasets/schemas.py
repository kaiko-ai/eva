"""Schema definitions for dataset classes."""

import dataclasses
from typing import Callable

from eva.language.data.datasets import schemas as language_schemas


@dataclasses.dataclass(frozen=True)
class TransformsSchema(language_schemas.TransformsSchema):
    """Schema for dataset transforms."""

    image: Callable | None = None
    """Image transformation"""
