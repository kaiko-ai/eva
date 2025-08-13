"""Data loader collator functions for the LLM multimodal data."""

from typing import List

from torch.utils.data._utils.collate import default_collate

from eva.multimodal.data.datasets.typings import TextImageSample
from eva.multimodal.models.typings import TextImageBatch


def text_image_collate(batch: List[TextImageSample]) -> TextImageBatch:
    return TextImageBatch(
        text=[sample.text for sample in batch],
        image=[sample.image for sample in batch],
        target=default_collate([sample.target for sample in batch]),
        metadata=(
            {k: [sample.metadata[k] for sample in batch] for k in batch[0].metadata.keys()}
            if batch[0].metadata
            else None
        ),
    )
