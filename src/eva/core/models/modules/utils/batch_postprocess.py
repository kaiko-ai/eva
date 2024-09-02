"""Batch post-processes module."""

import dataclasses
import functools
from typing import Any, Callable, Dict, List

import torch
from jsonargparse import _util
from lightning.pytorch.utilities.types import STEP_OUTPUT

Transform = Callable[[torch.Tensor], torch.Tensor]
"""Post-process transform type."""


@dataclasses.dataclass(frozen=True)
class BatchPostProcess:
    """Batch post-processes transform schema."""

    targets_transforms: List[Transform | Dict[str, Any]] | None = None
    """Holds the common train and evaluation metrics."""

    predictions_transforms: List[Transform | Dict[str, Any]] | None = None
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
                outputs["targets"], transforms=_parse_callable_inputs(self.targets_transforms)
            )

        if "predictions" in outputs and self.predictions_transforms is not None:
            outputs["predictions"] = _apply_transforms(
                outputs["predictions"],
                transforms=_parse_callable_inputs(self.predictions_transforms),
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


def _parse_callable_inputs(inputs: List[Callable | Dict[str, Any]]) -> List[Callable]:
    """Parses the inputs which where passed as dictionary to callable objects."""
    parsed = []
    for item in inputs:
        if isinstance(item, dict):
            item = _parse_dict(item)
        parsed.append(item)
    return parsed


def _parse_dict(item: Dict[str, Any]) -> Callable:
    """Parses the input dictionary to a partial callable object."""
    if not _is_valid_dict(item):
        raise ValueError(
            "Transform dictionary format is not valid. "
            "It must contain a key 'class_path' and optionally 'init_args' for "
            "the function and additional call arguments."
        )

    return functools.partial(
        _util.import_object(item["class_path"]),
        **item.get("init_args", {}),
    )


def _is_valid_dict(item: Dict[str, Any], /) -> bool:
    """Checks if the input has the valid structure."""
    return "class_path" in item and set(item.keys()) <= {"class_path", "init_args"}
