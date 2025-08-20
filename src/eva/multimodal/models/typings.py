"""Type definitions for multimodal models."""

from typing import Any, Dict, Generic, List, TypeVar

from torchvision import tv_tensors
from typing_extensions import NamedTuple

from eva.language.data.messages import MessageSeries

TargetType = TypeVar("TargetType")
"""The target data type."""


class TextImageBatch(NamedTuple, Generic[TargetType]):
    """Text and image sample with target and metadata."""

    text: List[MessageSeries]
    """A batch of conversations with one or multiple messages each."""

    image: List[tv_tensors.Image]
    """Image tensor."""

    target: TargetType | None
    """Target data."""

    metadata: Dict[str, Any] | None
    """Additional metadata."""
