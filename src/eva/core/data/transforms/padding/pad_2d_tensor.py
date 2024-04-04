"""Padding transformation for 2D tensors."""

import torch
import torch.nn.functional


class Pad2DTensor:
    """Pads a 2D tensor to a fixed dimension accross the first dimension."""

    def __init__(self, pad_size: int, pad_value: int | float = float("-inf")):
        """Initialize the transformation.

        Args:
            pad_size: The size to pad the tensor to. If the tensor is larger than this size,
                no padding will be applied.
            pad_value: The value to use for padding.
        """
        self._pad_size = pad_size
        self._pad_value = pad_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The input tensor of shape [n, embedding_dim].

        Returns:
            A tensor of shape [max(n, pad_dim), embedding_dim].
        """
        n_pad_values = self._pad_size - tensor.size(0)
        if n_pad_values > 0:
            tensor = torch.nn.functional.pad(
                tensor,
                pad=(0, 0, 0, n_pad_values),
                mode="constant",
                value=self._pad_value,
            )
        return tensor
