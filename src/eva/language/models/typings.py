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


class PredictionBatch(NamedTuple, Generic[TargetType]):
    """Text sample with target and metadata."""

    prediction: TargetType
    """Prediction data."""

    target: TargetType
    """Target data."""

    text: List[MessageSeries] | None
    """Conversation messages that were used as input."""

    metadata: Dict[str, Any] | None
    """Additional metadata."""
