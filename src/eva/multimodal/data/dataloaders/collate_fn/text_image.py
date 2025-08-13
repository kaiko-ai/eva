"""Data loader collator functions for the LLM multimodal data."""

from typing import Any, List, cast

from torch.utils.data._utils.collate import default_collate

from eva.multimodal.data.datasets.typings import TextImageSample
from eva.multimodal.models.typings import TextImageBatch


def text_image_collate(batch: List[TextImageSample]) -> TextImageBatch:
    """Collate function for text-image batches."""
    targets = [sample.target for sample in batch]
    collated_targets = cast(List[Any], default_collate(targets))

    first_sample = batch[0]
    metadata = None
    if first_sample.metadata is not None:
        metadata = {
            k: [sample.metadata[k] for sample in batch if sample.metadata]
            for k in first_sample.metadata.keys()
        }

    return TextImageBatch(
        text=[sample.text for sample in batch],
        image=[sample.image for sample in batch],
        target=collated_targets,
        metadata=metadata,
    )
