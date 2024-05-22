"""Intensity level scaling transform."""

import functools
from typing import Any, Dict, Tuple

import torch
import torchvision.transforms.v2 as torch_transforms
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.transforms.normalization import functional


class RescaleIntensity(torch_transforms.Transform):
    """Stretches or shrinks the image intensity levels."""

    def __init__(
        self,
        in_range: Tuple[float, float] | None = None,
        out_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Initializes the transform.

        Args:
            in_range: The input data range. If `None`, it will
                fetch the min and max of the input image.
            out_range: The desired intensity range of the output.
        """
        super().__init__()

        self._in_range = in_range
        self._out_range = out_range

    @functools.singledispatchmethod
    @override
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

    @_transform.register(torch.Tensor)
    def _(self, inpt: torch.Tensor, params: Dict[str, Any]) -> Any:
        return functional.rescale_intensity(
            inpt, in_range=self._in_range, out_range=self._out_range
        )

    @_transform.register(tv_tensors.Image)
    def _(self, inpt: tv_tensors.Image, params: Dict[str, Any]) -> Any:
        scaled_inpt = functional.rescale_intensity(inpt, out_range=self._out_range)
        return tv_tensors.wrap(scaled_inpt, like=inpt)

    @_transform.register(tv_tensors.BoundingBoxes)
    @_transform.register(tv_tensors.Mask)
    def _(self, inpt: tv_tensors.BoundingBoxes | tv_tensors.Mask, params: Dict[str, Any]) -> Any:
        return inpt
