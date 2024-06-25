"""Image clamp transform."""

import functools
from typing import Any, Dict, Tuple

import torch
import torchvision.transforms.v2 as torch_transforms
from torchvision import tv_tensors
from typing_extensions import override


class Clamp(torch_transforms.Transform):
    """Clamps all elements in input into a specific range."""

    def __init__(self, out_range: Tuple[int, int]) -> None:
        """Initializes the transform.

        Args:
            out_range: The lower and upper bound of the range to
                be clamped to.
        """
        super().__init__()

        self._out_range = out_range

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(torch.Tensor)
    def _(self, inpt: torch.Tensor, params: Dict[str, Any]) -> Any:
        return torch.clamp(inpt, min=self._out_range[0], max=self._out_range[1])

    @_transform.register(tv_tensors.Image)
    def _(self, inpt: tv_tensors.Image, params: Dict[str, Any]) -> Any:
        inpt_clamp = torch.clamp(inpt, min=self._out_range[0], max=self._out_range[1])
        return tv_tensors.wrap(inpt_clamp, like=inpt)

    @_transform.register(tv_tensors.BoundingBoxes)
    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: tv_tensors.BoundingBoxes | tv_tensors.Mask, params: Dict[str, Any]) -> Any:
        return inpt
