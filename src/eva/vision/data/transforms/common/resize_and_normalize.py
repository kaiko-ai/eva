"""Resizes and normalizes the input image."""

from typing import Callable, Sequence

import torch
import torchvision.transforms.v2 as torch_transforms

from eva.vision.data.transforms import structs
from src.eva.vision.data.transforms.defaults import DEFAULT_IMAGE_NORMALIZATION


class ResizeAndNormalize(torch_transforms.Compose):
    """Resizes, crops and normalizes an input image while preserving its aspect ratio."""

    def __init__(
        self,
        size: int | Sequence[int] = 224,
        normalize: structs.Normalize = DEFAULT_IMAGE_NORMALIZATION,
    ) -> None:
        """Initializes the transform object.

        Args:
            size: Desired output size of the crop. If size is an `int` instead
                of sequence like (h, w), a square crop (size, size) is made.
                Defaults to `224`.
            normalize: Normalize the output image with the specified mean and std.
                Defaults to `mean=(0.5, 0.5, 0.5)` and `std=(0.5, 0.5, 0.5)`.
        """
        self._size = size
        self._normalize = normalize
        super().__init__(transforms=self._build_transforms())

    def _build_transforms(self) -> Sequence[Callable]:
        """Builds and returns the list of transforms."""
        transforms = [
            torch_transforms.ToImage(),
            torch_transforms.Resize(size=self._size),
            torch_transforms.CenterCrop(size=self._size),
            torch_transforms.ToDtype(torch.float32, scale=True),
            torch_transforms.Normalize(
                mean=self._normalize.mean,
                std=self._normalize.std,
            ),
        ]
        return transforms
