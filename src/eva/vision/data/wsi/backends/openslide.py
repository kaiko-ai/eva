"""Module for loading data from WSI files using the OpenSlide library."""

from typing import Sequence, Tuple

import numpy as np
import openslide
from typing_extensions import override

from eva.vision.data.wsi.backends import base


class WsiOpenslide(base.Wsi):
    """Class for loading data from WSI files using the OpenSlide library."""

    _wsi: openslide.OpenSlide

    @override
    def open_file(self, file_path: str) -> openslide.OpenSlide:
        return openslide.OpenSlide(file_path)

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
        if self._wsi.properties.get(openslide.PROPERTY_NAME_MPP_X) and self._wsi.properties.get(
            openslide.PROPERTY_NAME_MPP_Y
        ):
            x_mpp = float(self._wsi.properties[openslide.PROPERTY_NAME_MPP_X])
            y_mpp = float(self._wsi.properties[openslide.PROPERTY_NAME_MPP_Y])
        elif (
            self._wsi.properties.get("tiff.XResolution")
            and self._wsi.properties.get("tiff.YResolution")
            and self._wsi.properties.get("tiff.ResolutionUnit")
        ):
            unit = self._wsi.properties.get("tiff.ResolutionUnit")
            if unit not in _conversion_factor_to_micrometer:
                raise ValueError(f"Unit {unit} not supported.")

            conversion_factor = float(_conversion_factor_to_micrometer.get(unit))  # type: ignore
            x_mpp = conversion_factor / float(self._wsi.properties["tiff.XResolution"])
            y_mpp = conversion_factor / float(self._wsi.properties["tiff.YResolution"])
        else:
            raise ValueError("`mpp` cannot be obtained for this slide.")

        return (x_mpp + y_mpp) / 2.0

    @override
    def _read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        return np.array(self._wsi.read_region(location, level, size))


_conversion_factor_to_micrometer = {
    "meter": 10**6,
    "decimeter": 10**5,
    "centimeter": 10**4,
    "millimeter": 10**3,
    "micrometer": 1,
    "nanometer": 10**-3,
    "picometer": 10**-6,
    "femtometer": 10**-9,
}
