"""Module for loading data from standard image file formats PIL library."""

from typing import Sequence, Tuple

import numpy as np
import PIL.Image
from typing_extensions import override

from eva.vision.data.wsi.backends import base


class PILImage(base.Wsi):
    """Class for loading data from standard image file formats using PIL library."""

    _wsi: PIL.Image.Image

    @override
    def open_file(self, file_path: str) -> PIL.Image.Image:
        return PIL.Image.open(file_path).convert("RGB")

    @property
    @override
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        return [self._wsi.size]

    @property
    @override
    def level_downsamples(self) -> Sequence[float]:
        return [1.0]

    @property
    @override
    def mpp(self) -> float:
        if self._overwrite_mpp is None:
            raise ValueError("Please specify the mpp using the `overwrite_mpp` argument.")
        return self._overwrite_mpp

    @override
    def _read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        width, height = size[0], size[1]
        patch = self._wsi.crop(
            # (left, upper, right, lower)
            (
                location[0],
                location[1],
                location[0] + width,
                location[1] + height,
            )
        )
        return np.array(patch)

    @override
    def _verify_location(self, location: Tuple[int, int], size: Tuple[int, int]) -> None:
        """Verifies that the requested coordinates are within the slide dimensions.

        Args:
            location: Top-left corner (x, y) coordinates to read.
            size: Size of the requested region (width, height)
        """
        x_max, y_max = self.level_dimensions[0]

        if int(location[0]) >= x_max or int(location[1]) >= y_max:
            raise ValueError(f"Out of bounds region: {location}, {size}")
