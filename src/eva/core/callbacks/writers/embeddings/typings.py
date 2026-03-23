"""Typing definitions for the writer callback functions."""

import dataclasses
import io
from typing import Any, Dict, List, NamedTuple


class QUEUE_ITEM(NamedTuple):
    """The default input batch data scheme."""

    prediction_buffer: io.BytesIO
    """IO buffer containing the prediction tensor."""

    target_buffer: io.BytesIO
    """IO buffer containing the target tensor."""

    data_name: str
    """Name of the input data that was used to generate the embedding."""

    save_name: str
    """Name to store the generated embedding."""

    split: str | None
    """The dataset split the item belongs to (e.g. train, val, test)."""

    metadata: Dict[str, Any] | None = None
    """Dictionary holding additional metadata."""


@dataclasses.dataclass
class ITEM_DICT_ENTRY:
    """Typing for holding queue items and number of save operations."""

    items: List[QUEUE_ITEM]
    """List of queue items."""

    save_count: int
    """Number of prior item batch saves to same file."""
