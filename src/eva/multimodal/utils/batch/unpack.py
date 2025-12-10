"""Unpack batch utility function."""

from typing import Any, Dict, List, Tuple

from torchvision import tv_tensors

from eva.language.data.messages import MessageSeries
from eva.language.models.typings import TextBatch
from eva.multimodal.models.typings import TextImageBatch


def unpack_batch(
    batch: TextImageBatch | TextBatch,
) -> Tuple[
    List[MessageSeries],
    List[List[tv_tensors.Image]] | None,
    Any,
    Dict[str, Any] | None,
]:
    """Unpacks a TextImageBatch or TextBatch into its components."""
    if isinstance(batch, TextImageBatch):
        return batch.text, batch.images, batch.target, batch.metadata
    return batch.text, None, batch.target, batch.metadata
