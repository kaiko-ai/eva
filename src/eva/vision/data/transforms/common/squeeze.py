"""Squeeze transform."""

from typing import Any

import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.transforms import base


class Squeeze(base.TorchvisionTransformV2):
    """Squeezes the input tensor accross all or specified dimensions."""

    def __init__(self, dim: int | list[int] | None = None):
        """Initializes the transform.

        Args:
            dim: If specified, the input will be squeezed only in the specified dimensions.
        """
        super().__init__()
        self._dim = dim

    @override
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        output = torch.squeeze(inpt) if self._dim is None else torch.squeeze(inpt, dim=self._dim)
        return tv_tensors.wrap(output, like=inpt)
