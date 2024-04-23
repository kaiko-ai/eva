"""Module for loading data from WSI files using the OpenSlide library."""

from typing import Sequence, Tuple

import numpy as np
import openslide
from typing_extensions import override

from eva.vision.data.wsi import base


class WsiOpenslide(base.Wsi):
    """Class for loading data from WSI files using the OpenSlide library."""

    _wsi: openslide.OpenSlide | openslide.ImageSlide

    @override
    def open_file(self, file_path: str) -> openslide.OpenSlide | openslide.ImageSlide:
        return openslide.open_slide(file_path)

    @property
    @override
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        return self._wsi.level_dimensions

    @property
    @override
    def level_downsamples(self) -> Sequence[float]:
        return self._wsi.level_downsamples

    @property
    @override
    def mpp(self) -> float:
        # TODO: add overwrite_mpp class attribute to allow setting a default value
        x_mpp = float(self._wsi.properties["openslide.mpp-x"])
        y_mpp = float(self._wsi.properties["openslide.mpp-y"])
        return (x_mpp + y_mpp) / 2.0

    @override
    def read_region(
        self, location: Tuple[int, int], size: Tuple[int, int], level: int
    ) -> np.ndarray:
        x_max, y_max = self._wsi.level_dimensions[level]
        if location[0] + size[0] > x_max or location[1] + size[1] > y_max:
            raise ValueError(f"Out of bounds region: {location}, {size}, {level}")
        data = self._wsi.read_region(location, level, size)
        return np.array(data.convert("RGB"))
