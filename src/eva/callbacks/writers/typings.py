"""Typing definitions for the writer callback functions."""

import io
from typing import NamedTuple


class QUEUE_ITEM(NamedTuple):
    """The default input batch data scheme."""

    prediction_buffer: io.BytesIO
    """IO buffer containing the prediction tensor"""

    target_buffer: io.BytesIO
    """IO buffer containing the target tensor"""

    input_name: str
    """Name of the original input file that was used to generate the embedding."""

    save_name: str
    """Name to store the generated embedding"""

    split: str | None
    """The dataset split the item belongs to (e.g. train, val, test)."""
