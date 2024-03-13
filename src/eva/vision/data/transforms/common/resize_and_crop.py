"""Resizes and normalizes the input image."""

from typing import Callable, Sequence

import torch
import torchvision.transforms.v2 as torch_transforms


class ResizeAndCrop(torch_transforms.Compose):
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
            torch_transforms.ToImage(),
            torch_transforms.Resize(size=self._size),
            torch_transforms.CenterCrop(size=self._size),
            torch_transforms.ToDtype(torch.float32, scale=True),
            torch_transforms.Normalize(
                mean=self._mean,
                std=self._std,
            ),
        ]
        return transforms
