"""Specialized transforms for resizing, clamping and range normalizing."""

from typing import Callable, Sequence, Tuple

from torchvision.transforms import v2

from eva.vision.data.transforms import normalization


class ResizeAndClamp(v2.Compose):
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
            v2.Resize(size=self._size),
            v2.CenterCrop(size=self._size),
            normalization.Clamp(out_range=self._clamp_range),
            normalization.RescaleIntensity(
                in_range=self._clamp_range,
                out_range=(0.0, 1.0),
            ),
            v2.Normalize(
                mean=self._mean,
                std=self._std,
            ),
        ]
        return transforms
