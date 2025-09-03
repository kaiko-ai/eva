"""Collate functions for text data."""

from typing import List

from torch.utils.data._utils.collate import default_collate

from eva.language.data.datasets.typings import PredictionSample, TextSample
from eva.language.models.typings import PredictionBatch, TextBatch


def text_collate(batch: List[TextSample]) -> TextBatch:
    """Collate function for text data that keeps texts as separate strings.

    Args:
        batch: List of tuples containing (text, target, metadata) from the dataset

    Returns:
        A batch of text samples with targets and metadata.
    """
    texts, targets, metadata = zip(*batch, strict=False)
    first_sample = batch[0]
    metadata = None
    if first_sample.metadata is not None:
        metadata = {
            k: [sample.metadata[k] for sample in batch if sample.metadata]
            for k in first_sample.metadata.keys()
        }
    return TextBatch(
        text=list(texts),
        target=default_collate(targets) if targets[0] is not None else None,
        metadata=metadata,
    )


def prediction_collate(batch: List[PredictionSample]) -> PredictionBatch:
    """Collate function for text prediction data.

    Args:
        batch: List of tuples containing (prediction, target, text, metadata) from the dataset

    Returns:
        A batch of prediction samples.
    """
    predictions, targets, texts, metadata = zip(*batch, strict=False)
    first_sample = batch[0]
    metadata = None
    if first_sample.metadata is not None:
        metadata = {
            k: [sample.metadata[k] for sample in batch if sample.metadata]
            for k in first_sample.metadata.keys()
        }
    return PredictionBatch(
        prediction=default_collate(predictions) if predictions[0] is not None else None,
        target=default_collate(targets) if targets[0] is not None else None,
        text=list(texts) if first_sample.text is not None else None,
        metadata=metadata,
    )
