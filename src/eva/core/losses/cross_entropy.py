"""Cross-entropy based loss function."""

from typing import Sequence
from typing_extensions import override

import torch
from torch import nn




class CrossEntropyLoss(nn.CrossEntropyLoss):
    """A wrapper around torch.nn.CrossEntropyLoss that accepts weights in list format.

    Needed for .yaml file loading & class instantiation with jsonarparse.
    """

    def __init__(
        self, *args, weight: Sequence[float] | torch.Tensor | None = None, **kwargs
    ) -> None:
        """Initialize the loss function.

        Args:
            args: Positional arguments from the base class.
            weight: A list of weights to assign to each class.
            kwargs: Key-word arguments from the base class.
        """
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        super().__init__(*args, **kwargs, weight=weight)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss with label smoothing."""
    def __init__(self, *args, label_smoothing: float=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing value must be between 0 and 1.")
        self.label_smoothing = label_smoothing

    @override
    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

        loss = super().forward(input, target)
        return loss