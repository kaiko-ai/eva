"""Batch post-processes module."""

import dataclasses
import functools
from typing import Any, Callable, Dict, List

import torch
from jsonargparse import _util
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

Transform = Callable[[torch.Tensor], torch.Tensor]
"""Post-process transform type."""


@dataclasses.dataclass(frozen=True)
class BatchPostProcess:
    """Batch post-processes transform schema."""

    targets_transforms: List[Transform | Dict[str, Any]] | None = None
    """Holds the common train and evaluation metrics."""

    predictions_transforms: List[Transform | Dict[str, Any]] | None = None
    """Holds the common train and evaluation metrics."""

    @override
    def __post_init__(self) -> None:
        self._parse_transforms(self.targets_transforms or [])
        self._parse_transforms(self.predictions_transforms or [])

    def _parse_transforms(self, inputs: List[Transform]) -> None:
        """Parses in-place transforms which where passed as functions."""
        for i, transform in enumerate(inputs):
            if isinstance(transform, dict):
                inputs[i] = functools.partial(
                    _util.import_object(transform["class_path"]), **transform.get("init_args", {})
                )

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
