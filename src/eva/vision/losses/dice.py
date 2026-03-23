"""Dice based loss functions."""

from typing import Sequence, Tuple

import torch
from monai import losses
from monai.networks import one_hot  # type: ignore
from typing_extensions import override


class DiceLoss(losses.DiceLoss):  # type: ignore
    """Computes the average Dice loss between two tensors.

    Extends the implementation from MONAI
        - to support semantic target labels (meaning targets of shape BHW)
        - to support `ignore_index` functionality
        - accept weight argument in list format
    """

    def __init__(
        self,
        *args,
        ignore_index: int | None = None,
        weight: Sequence[float] | torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Initialize the DiceLoss.

        Args:
            args: Positional arguments from the base class.
            ignore_index: Specifies a target value that is ignored and
                does not contribute to the input gradient.
            weight: A list of weights to assign to each class.
            kwargs: Key-word arguments from the base class.
        """
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)

        super().__init__(*args, **kwargs, weight=weight)

        self.ignore_index = ignore_index

    @override
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa
        inputs, targets = _apply_ignore_index(inputs, targets, self.ignore_index)
        targets = _to_one_hot(targets, num_classes=inputs.shape[1])

        return super().forward(inputs, targets)


class DiceCELoss(losses.dice.DiceCELoss):
    """Combination of Dice and Cross Entropy Loss.

    Extends the implementation from MONAI
        - to support semantic target labels (meaning targets of shape BHW)
        - to support `ignore_index` functionality
        - accept weight argument in list format
    """

    def __init__(
        self,
        *args,
        ignore_index: int | None = None,
        weight: Sequence[float] | torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Initialize the DiceCELoss.

        Args:
            args: Positional arguments from the base class.
            ignore_index: Specifies a target value that is ignored and
                does not contribute to the input gradient.
            weight: A list of weights to assign to each class.
            kwargs: Key-word arguments from the base class.
        """
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)

        super().__init__(*args, **kwargs, weight=weight)

        self.ignore_index = ignore_index

    @override
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa
        inputs, targets = _apply_ignore_index(inputs, targets, self.ignore_index)
        targets = _to_one_hot(targets, num_classes=inputs.shape[1])

        return super().forward(inputs, targets)


def _apply_ignore_index(
    inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int | None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if ignore_index is not None:
        mask = targets != ignore_index
        targets = targets * mask
        inputs = torch.mul(inputs, mask.unsqueeze(1) if mask.ndim == 3 else mask)
    return inputs, targets


def _to_one_hot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    if tensor.ndim == 3:
        return one_hot(tensor[:, None, ...], num_classes=num_classes)
    return tensor
