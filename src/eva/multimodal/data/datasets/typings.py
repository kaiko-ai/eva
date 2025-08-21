"""Typings for multimodal datasets."""

from typing import Any, Generic, TypeVar

from torchvision import tv_tensors
from typing_extensions import NamedTuple

from eva.language.data.messages import MessageSeries

TargetType = TypeVar("TargetType")
"""The target data type."""


class TextImageSample(NamedTuple, Generic[TargetType]):
    """Text and image sample with target and metadata."""

    text: MessageSeries
    """One or multiple conversation messages."""

    image: tv_tensors.Image
    """Image tensor."""

    target: TargetType | None
    """Target data."""

    metadata: dict[str, Any] | None
    """Additional metadata."""
