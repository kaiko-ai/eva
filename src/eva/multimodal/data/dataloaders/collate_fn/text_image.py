"""Collate functions for text-image data."""

from typing import List

from torch.utils.data._utils.collate import default_collate

from eva.multimodal.data.datasets.typings import TextImageSample
from eva.multimodal.models.typings import TextImageBatch


def text_image_collate(batch: List[TextImageSample]) -> TextImageBatch:
    """Collate function for text-image batches."""
    texts, images, targets, metadata = zip(*batch, strict=False)

    first_sample = batch[0]
    metadata = None
    if first_sample.metadata is not None:
        metadata = {
            k: [sample.metadata[k] for sample in batch if sample.metadata]
            for k in first_sample.metadata.keys()
        }

    return TextImageBatch(
        text=list(texts),
        image=list(images),
        target=default_collate(targets) if targets[0] is not None else None,
        metadata=metadata,
    )
