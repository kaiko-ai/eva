from typing import Any, Dict, Generic, List, Literal, NamedTuple, TypeVar

from eva.language.data.messages import MessageSeries

TargetType = TypeVar("TargetType")
"""The target data type."""


class TextBatch(NamedTuple, Generic[TargetType]):
    """Text sample with target and metadata."""

    text: List[MessageSeries]
    """Text content."""

    target: List[TargetType | None]
    """Target data."""

    metadata: Dict[str, Any] | None
    """Additional metadata."""


ModelType = Literal["huggingface", "internal", "api"]
"""The type of model being used."""
