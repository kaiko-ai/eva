"""Clone related functions."""

import functools
from typing import Any, Dict, List

import torch


@functools.singledispatch
def clone(tensor_type: Any) -> Any:
    """Clone tensor objects."""
    raise TypeError(f"Unsupported input type: {type(input)}.")


@clone.register
def _(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()


@clone.register
def _(tensors: list) -> List[torch.Tensor]:
    return list(map(clone, tensors))


@clone.register
def _(tensors: dict) -> Dict[str, torch.Tensor]:
    return {key: clone(tensors[key]) for key in tensors}
