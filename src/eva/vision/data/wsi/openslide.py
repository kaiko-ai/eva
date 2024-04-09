from typing import Any

import openslide
from typing_extensions import override

from eva.vision.data import wsi


class WsiOpenslide(wsi.Wsi):
    _slide: openslide.OpenSlide

    @override
    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return self._slide.level_dimensions

    @override
    @property
    def level_downsamples(self) -> list[float]:
        return self._slide.level_downsamples

    @override
    @property
    def mpp(self) -> float:
        try:
            x_mpp = float(self._slide.properties["openslide.mpp-x"])
            y_mpp = float(self._slide.properties["openslide.mpp-y"])
            return (x_mpp + y_mpp) / 2.0
        except KeyError:
            # TODO: add overwrite_mpp class attribute to allow setting a default value
            raise ValueError("Microns per pixel (mpp) value is not available for this slide.")

    @override
    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Any:
        return self._slide.read_region(location, level, size)

    @override
    @staticmethod
    def open_slide(file_path: str) -> openslide.OpenSlide:
        return openslide.OpenSlide(file_path)
