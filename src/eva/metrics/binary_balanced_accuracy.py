"""Implementation of the balanced accuracy for binary classification tasks."""

import torch
from torch import Tensor
from torchmetrics import classification
from typing_extensions import override


class BinaryBalancedAccuracy(classification.MulticlassAccuracy):
    """Computes the balanced accuracy for binary classification.

    Currently, torchmetrics does not support balanced accuracy for binary
    classification tasks (see https://github.com/Lightning-AI/torchmetrics/issues/1706)

    This class is a workaround that uses the `MulticlassAccuracy` class with
    `num_classes=2` and `average="macro"` to compute the balanced accuracy for
    binary classification tasks.
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int | None = None) -> None:
        """Initializes the binary balanced accuracy metric.

        Args:
            threshold: Threshold for transforming probability to binary (0,1) predictions
            ignore_index: Specifies a target value that is ignored and does not
                contribute to the metric calculation.

        """
        super().__init__(average="macro", num_classes=2, ignore_index=ignore_index)

        self._threshold = threshold

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds > self._threshold
        super().update(preds, target.to(torch.int64))
