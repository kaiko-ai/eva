"""Transformations to change the shape of tensors."""

import torch


class SqueezeTensor:
    """Squeezes a [B, 1] tensor to [B]."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The input tensor to be squeezed.
        """
        return tensor.squeeze(-1)
