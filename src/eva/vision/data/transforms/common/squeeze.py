"""Squeeze transform."""

from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


class Squeeze(v2.Transform):
    """Squeezes the input tensor accross all or specified dimensions."""

    def __init__(self, dim: int | list[int] | None = None):
        """Initializes the transform.

        Args:
            dim: If specified, the input will be squeezed only in the specified dimensions.
        """
        super().__init__()
        self._dim = dim

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        output = torch.squeeze(inpt) if self._dim is None else torch.squeeze(inpt, dim=self._dim)
        return tv_tensors.wrap(output, like=inpt)
