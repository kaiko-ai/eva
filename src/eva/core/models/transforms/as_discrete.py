"""Defines the AsDiscrete transformation."""

import torch


class AsDiscrete:
    """Convert the logits tensor to discrete values."""

    def __init__(
        self,
        argmax: bool = False,
        to_onehot: int | bool | None = None,
        threshold: float | None = None,
        dim: int = 1,
        keepdim: bool = True,
    ) -> None:
        """Convert the input tensor/array into discrete values.

        Args:
            argmax: Whether to execute argmax function on input data before transform.
            to_onehot: if not None, convert input data into the one-hot format with
                specified number of classes. If bool, it will try to infer the number
                of classes.
            threshold: If not None, threshold the float values to int number 0 or 1
                with specified threshold.
            dim: The dimension along which to apply argmax and/or one-hot encoding.
            keepdim: Whether to keep the reduced dimension after argmax.
        """
        super().__init__()

        self._argmax = argmax
        self._dim = dim
        self._keepdim = keepdim
        self._to_onehot = to_onehot
        self._threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Call method for the transformation."""
        if self._argmax:
            tensor = torch.argmax(tensor, dim=self._dim, keepdim=self._keepdim)

        if self._to_onehot is not None:
            tensor = _one_hot(tensor, num_classes=self._to_onehot, dim=self._dim, dtype=torch.long)

        if self._threshold is not None:
            tensor = tensor >= self._threshold

        return tensor


def _one_hot(
    tensor: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1
) -> torch.Tensor:
    """Convert input tensor into one-hot format (implementation taken from MONAI)."""
    shape = list(tensor.shape)
    if shape[dim] != 1:
        raise AssertionError(f"Input tensor must have 1 channel at dim {dim}.")

    shape[dim] = num_classes
    o = torch.zeros(size=shape, dtype=dtype, device=tensor.device)
    tensor = o.scatter_(dim=dim, index=tensor.long(), value=1)

    return tensor
