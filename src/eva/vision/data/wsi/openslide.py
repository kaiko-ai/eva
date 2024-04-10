from typing import List, Tuple

import numpy as np
import openslide
from typing_extensions import override

from eva.vision.data.wsi import base


class WsiOpenslide(base.Wsi):
    """Class for loading data from WSI files using the OpenSlide library."""

    _wsi: openslide.OpenSlide

    @override
    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        return self._wsi.level_dimensions

    @override
    @property
    def level_downsamples(self) -> List[float]:
        return self._wsi.level_downsamples

    @override
    @property
    def mpp(self) -> float:
        try:
            x_mpp = float(self._wsi.properties["openslide.mpp-x"])
            y_mpp = float(self._wsi.properties["openslide.mpp-y"])
            return (x_mpp + y_mpp) / 2.0
        except KeyError:
            # TODO: add overwrite_mpp class attribute to allow setting a default value
            raise ValueError("Microns per pixel (mpp) value is not available for this slide.")

    @override
    def read_region(
        self, location: Tuple[int, int], size: Tuple[int, int], level: int
    ) -> np.ndarray:
        data = self._wsi.read_region(location, level, size)

        return np.array(data.convert("RGB"))

    @override
    def open_slide(self) -> openslide.OpenSlide:
        self._wsi = openslide.open_slide(self._file_path)
