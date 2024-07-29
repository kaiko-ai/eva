"""Dice loss."""

import torch
from monai import losses
from monai.networks import one_hot  # type: ignore
from typing_extensions import override


class DiceLoss(losses.DiceLoss):  # type: ignore
    """Computes the average Dice loss between two tensors.

    Extends the implementation from MONAI
        - to support semantic target labels (meaning targets of shape BHW)
        - to support `ignore_index` functionality
    """

    def __init__(self, *args, ignore_index: int | None = None, **kwargs) -> None:
        """Initialize the DiceLoss with support for ignore_index.

        Args:
            args: Positional arguments from the base class.
            ignore_index: Specifies a target value that is ignored and
                does not contribute to the input gradient.
            kwargs: Key-word arguments from the base class.
        """
        super().__init__(*args, **kwargs)

        self.ignore_index = ignore_index

    @override
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            targets = targets * mask
            inputs = torch.mul(inputs, mask.unsqueeze(1) if mask.ndim == 3 else mask)

        if targets.ndim == 3:
            targets = one_hot(targets[:, None, ...], num_classes=inputs.shape[1])

        return super().forward(inputs, targets)
