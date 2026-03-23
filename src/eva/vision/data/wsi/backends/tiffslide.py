"""Module for loading data from WSI files using the OpenSlide library."""

from typing import Sequence, Tuple

import numpy as np
import tiffslide  # type: ignore
from typing_extensions import override

from eva.vision.data.wsi.backends import base


class WsiTiffslide(base.Wsi):
    """Class for loading data from WSI files using the TiffSlide library."""

    _wsi: tiffslide.TiffSlide

    @override
    def open_file(self, file_path: str) -> tiffslide.TiffSlide:
        return tiffslide.TiffSlide(file_path)

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
        x_mpp = float(self._wsi.properties[tiffslide.PROPERTY_NAME_MPP_X])
        y_mpp = float(self._wsi.properties[tiffslide.PROPERTY_NAME_MPP_Y])
        return (x_mpp + y_mpp) / 2.0

    @override
    def _read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        return np.array(self._wsi.read_region(location, level, size))
