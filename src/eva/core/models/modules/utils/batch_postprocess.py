"""Batch post-processes module."""

import dataclasses
import functools
from typing import Callable, List

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

Transform = Callable[[torch.Tensor], torch.Tensor]
"""Post-process transform type."""


@dataclasses.dataclass(frozen=True)
class BatchPostProcess:
    """Batch post-processes transform schema."""

    targets_transforms: List[Transform] | None = None
    """Holds the common train and evaluation metrics."""

    predictions_transforms: List[Transform] | None = None
    """Holds the common train and evaluation metrics."""

    def __call__(self, outputs: STEP_OUTPUT) -> None:
        """Applies the defined list of transforms to the batch output in-place.

        Note that the transforms are applied only when the input is a dictionary
        and only to its keys of `predictions` and/or `targets`.

        Args:
            outputs: The batch output of the model module step.
        """
        if not isinstance(outputs, dict):
            return

        if "targets" in outputs and self.targets_transforms is not None:
            outputs["targets"] = _apply_transforms(
                outputs["targets"], transforms=self.targets_transforms
            )

        if "predictions" in outputs and self.predictions_transforms is not None:
            outputs["predictions"] = _apply_transforms(
                outputs["predictions"], transforms=self.predictions_transforms
            )


def _apply_transforms(tensor: torch.Tensor, transforms: List[Transform]) -> torch.Tensor:
    """Applies a list of transforms sequentially to a input tensor.

    Args:
        tensor: The desired tensor to process.
        transforms: The list of transforms to apply to the input tensor.

    Returns:
        The processed tensor.
    """
    return functools.reduce(lambda tensor, transform: transform(tensor), transforms, tensor)
