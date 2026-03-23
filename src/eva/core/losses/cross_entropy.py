"""Cross-entropy based loss function."""

from typing import Sequence

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
