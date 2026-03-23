"""Utils for segmentation metric collections."""

from typing import Tuple

import torch


def apply_ignore_index(
    preds: torch.Tensor, target: torch.Tensor, ignore_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies the ignore index to the predictions and target tensors.

    1. Masks the values in the target tensor that correspond to the ignored index.
    2. Remove the channel corresponding to the ignored index from both tensors.

    Args:
        preds: The predictions tensor. Expected to be of shape `(N,C,...)`.
        target: The target tensor. Expected to be of shape `(N,C,...)`.
        ignore_index: The index to ignore.

    Returns:
        The modified predictions and target tensors of shape `(N,C-1,...)`.
    """
    if ignore_index < 0:
        raise ValueError("ignore_index must be a non-negative integer")

    ignore_mask = preds[:, ignore_index] == 1
    target = target * (~ignore_mask.unsqueeze(1))

    preds = _ignore_tensor_channel(preds, ignore_index)
    target = _ignore_tensor_channel(target, ignore_index)

    return preds, target


def index_to_one_hot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts an index tensor to a one-hot tensor.

    Args:
        tensor: The index tensor to convert. Expected to be of shape `(N,...)`.
        num_classes: The number of classes to one-hot encode.

    Returns:
        A one-hot tensor of shape `(N,C,...)`.
    """
    if not _is_one_hot(tensor):
        tensor = torch.nn.functional.one_hot(tensor.long(), num_classes=num_classes).movedim(-1, 1)
    return tensor


def _ignore_tensor_channel(tensor: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Removes the channel corresponding to the specified ignore index.

    Args:
        tensor: The tensor to remove the channel from. Expected to be of shape `(N,C,...)`.
        ignore_index: The index of the channel dimension (C) to remove.

    Returns:
        A tensor without the specified channel `(N,C-1,...)`.
    """
    if ignore_index < 0:
        raise ValueError("ignore_index must be a non-negative integer")
    return torch.cat([tensor[:, :ignore_index], tensor[:, ignore_index + 1 :]], dim=1)


def _is_one_hot(tensor: torch.Tensor, expected_dim: int = 4) -> bool:
    """Checks if the tensor is a one-hot tensor."""
    return bool((tensor.bool() == tensor).all()) and tensor.ndim == expected_dim
