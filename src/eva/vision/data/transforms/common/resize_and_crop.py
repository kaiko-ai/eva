"""Resizes and normalizes the input image."""

from typing import Callable, Sequence

import torch
from torchvision.transforms import v2


class ResizeAndCrop(v2.Compose):
    """Resizes, crops and normalizes an input image while preserving its aspect ratio."""

    def __init__(
        self,
        size: int | Sequence[int] = 224,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        """Initializes the transform object.

        Args:
            size: Desired output size of the crop. If size is an `int` instead
                of sequence like (h, w), a square crop (size, size) is made.
            mean: Sequence of means for each image channel.
            std: Sequence of standard deviations for each image channel.
        """
        self._size = size
        self._mean = mean
        self._std = std

        super().__init__(transforms=self._build_transforms())

    def _build_transforms(self) -> Sequence[Callable]:
        """Builds and returns the list of transforms."""
        transforms = [
            v2.Resize(size=self._size),
            v2.CenterCrop(size=self._size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self._mean,
                std=self._std,
            ),
        ]
        return transforms
