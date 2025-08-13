"""Type definitions for multimodal models."""

from typing import Any, Dict, Generic, List, NamedTuple, TypedDict, TypeVar

from torchvision import tv_tensors
from typing_extensions import NotRequired

from eva.language.data.messages import MessageSeries

TargetType = TypeVar("TargetType")
"""The target data type."""


class TextImageBatch(NamedTuple, Generic[TargetType]):
    """Text and image sample with target and metadata."""

    text: List[MessageSeries]
    """A batch of conversations with one or multiple messages each."""

    image: List[tv_tensors.Image]
    """Image tensor."""

    target: List[TargetType | None]
    """Target data."""

    metadata: Dict[str, Any] | None
    """Additional metadata."""


class VisionLanguageOutput(TypedDict):
    """Output data schema for vision-language models."""

    output: List[str]
    """Generated text outputs."""

    processed_input: List[Any]
    """Inputs after preprocessing."""

    raw_input: NotRequired[List[str]]
    """Original inputs without processing."""

    metadata: NotRequired[Dict[str, Any]]
    """Additional metadata."""
