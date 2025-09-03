"""Typings for multimodal datasets."""

from typing import Any, Generic, TypeVar

from typing_extensions import NamedTuple

from eva.language.data.messages import MessageSeries

TargetType = TypeVar("TargetType")
"""The target data type."""


class TextSample(NamedTuple, Generic[TargetType]):
    """Text sample with target and metadata."""

    text: MessageSeries
    """One or multiple conversation messages."""

    target: TargetType | None
    """Target data."""

    metadata: dict[str, Any] | None
    """Additional metadata."""


class PredictionSample(NamedTuple, Generic[TargetType]):
    """Text sample with target and metadata."""

    prediction: TargetType
    """Prediction data."""

    target: TargetType
    """Target data."""

    text: MessageSeries | None
    """Conversation messages that were used as input."""

    metadata: dict[str, Any] | None
    """Additional metadata."""
