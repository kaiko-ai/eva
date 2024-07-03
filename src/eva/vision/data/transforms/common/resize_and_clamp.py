"""Specialized transforms for resizing, clamping and range normalizing."""

from typing import Callable, Sequence, Tuple

import torch
<<<<<<< HEAD
import torchvision.transforms.v2 as torch_transforms
=======
from torchvision.transforms import v2
>>>>>>> main

from eva.vision.data.transforms import normalization


<<<<<<< HEAD
class ResizeAndClamp(torch_transforms.Compose):
=======
class ResizeAndClamp(v2.Compose):
>>>>>>> main
    """Resizes, crops, clamps and normalizes an input image."""

    def __init__(
        self,
        size: int | Sequence[int] = 224,
        clamp_range: Tuple[int, int] = (-1024, 1024),
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        std: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Initializes the transform object.

        Args:
            size: Desired output size of the crop. If size is an `int` instead
                of sequence like (h, w), a square crop (size, size) is made.
            clamp_range: The lower and upper bound to clamp the pixel values.
            mean: Sequence of means for each image channel.
            std: Sequence of standard deviations for each image channel.
        """
        self._size = size
        self._clamp_range = clamp_range
        self._mean = mean
        self._std = std

        super().__init__(transforms=self._build_transforms())

    def _build_transforms(self) -> Sequence[Callable]:
        """Builds and returns the list of transforms."""
        transforms = [
<<<<<<< HEAD
            torch_transforms.Resize(size=self._size),
            torch_transforms.CenterCrop(size=self._size),
            normalization.Clamp(out_range=self._clamp_range),
            torch_transforms.ToDtype(torch.float32),
=======
            v2.Resize(size=self._size),
            v2.CenterCrop(size=self._size),
            normalization.Clamp(out_range=self._clamp_range),
            v2.ToDtype(torch.float32),
>>>>>>> main
            normalization.RescaleIntensity(
                in_range=self._clamp_range,
                out_range=(0.0, 1.0),
            ),
<<<<<<< HEAD
            torch_transforms.Normalize(
=======
            v2.Normalize(
>>>>>>> main
                mean=self._mean,
                std=self._std,
            ),
        ]
        return transforms
