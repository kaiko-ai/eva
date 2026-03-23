"""Type annotations for language model modules."""

from typing import Any, Dict, List, NamedTuple


class TEXT_BATCH(NamedTuple):
    """Text-based input batch data scheme."""

    data: List[str]
    """The text data batch."""

    targets: List[str] | None = None
    """The target text batch."""

    metadata: Dict[str, Any] | None = None
    """The associated metadata."""
