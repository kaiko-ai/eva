"""Unpack batch utility function."""

from eva.language.models.typings import TextBatch
from eva.multimodal.models.typings import TextImageBatch


def unpack_batch(batch: TextImageBatch | TextBatch) -> tuple:
    """Unpacks a TextImageBatch or TextBatch into its components."""
    if isinstance(batch, TextImageBatch):
        return batch.text, batch.image, batch.target, batch.metadata
    return batch.text, None, batch.target, batch.metadata
