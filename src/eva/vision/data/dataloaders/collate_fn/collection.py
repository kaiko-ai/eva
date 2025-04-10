"""Data only collate filter function."""

from typing import Any, List

import torch

from eva.core.models.modules.typings import INPUT_BATCH


def collection_collate(batch: List[List[INPUT_BATCH]]) -> Any:
    """Collate function for stacking a collection of data samples.

    Args:
        batch: The batch to be collated.

    Returns:
        The collated batch.
    """
    tensors, targets, metadata = zip(*batch, strict=False)
    batch_tensors = torch.cat(list(map(torch.stack, tensors)))
    batch_targets = torch.cat(list(map(torch.stack, targets)))
    return batch_tensors, batch_targets, metadata
