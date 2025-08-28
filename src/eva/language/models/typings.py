"""Type definitions for language models."""

from typing import Any, Dict, Generic, List, TypeVar

from typing_extensions import NamedTuple

from eva.language.data.messages import MessageSeries

TargetType = TypeVar("TargetType")
"""The target data type."""


class TextBatch(NamedTuple, Generic[TargetType]):
    """Text sample with target and metadata."""

    text: List[MessageSeries]
    """Text content."""

    target: TargetType | None
    """Target data."""

    metadata: Dict[str, Any] | None
    """Additional metadata."""
