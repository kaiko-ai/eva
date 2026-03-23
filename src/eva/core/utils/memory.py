"""Memory related functions."""

import functools
from typing import Any, Dict, List

import torch


@functools.singledispatch
def to_cpu(tensor_type: Any) -> Any:
    """Moves tensor objects to `cpu`."""
    raise TypeError(f"Unsupported input type: {type(tensor_type)}.")


@to_cpu.register
def _(tensor: torch.Tensor) -> torch.Tensor:
    detached_tensor = tensor.detach()
    return detached_tensor.cpu()


@to_cpu.register
def _(tensors: list) -> List[torch.Tensor]:
    return list(map(to_cpu, tensors))


@to_cpu.register
def _(tensors: dict) -> Dict[str, torch.Tensor]:
    return {key: to_cpu(tensors[key]) for key in tensors}
