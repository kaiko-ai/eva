"""Collate functions for text data."""

from typing import Dict, List, Tuple

import torch


def text_collate_fn(
    batch: List[Tuple[str, torch.Tensor, Dict]],
) -> Tuple[List[str], torch.Tensor, List[Dict]]:
    """Collate function for text data that keeps texts as separate strings.

    Args:
        batch: List of tuples containing (text, target, metadata) from the dataset

    Returns:
        Tuple containing:
            - List of text strings
            - Batched tensor of targets
            - List of metadata dictionaries
    """
    texts, targets, metadata = zip(*batch, strict=False)
    targets = torch.stack(targets)
    return list(texts), targets, list(metadata)
