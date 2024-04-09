import abc
from typing import Any, List, Tuple

import numpy as np


class Wsi(abc.ABC):
    def __init__(self, file_path: str):
        """Initializes a new class instance.

        Args:
            file_path: The path to the whole slide image file.
        """
        self._file_path = file_path
        self._wsi = None

    @property
    @abc.abstractmethod
    def level_dimensions(self) -> List[tuple[int, int]]:
        """A list of (width, height) tuples for each zoom level, from highest to lowest resolution."""

    @property
    @abc.abstractmethod
    def level_downsamples(self) -> List[float]:
        """A list of downsampling factors for each zoom level, relative to the highest resolution."""

    @property
    @abc.abstractmethod
    def mpp(self) -> float:
        """Microns per pixel at the highest resolution, indicating physical size per image pixel."""

    @abc.abstractmethod
    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Reads and returns image data for a specified region and zoom level.

        Args:
            location: Top-left corner (x, y) to start reading.
            level: Zoom level, with 0 being the highest resolution.
            size: Region size as (width, height).
        """

    @abc.abstractmethod
    def open_slide(self) -> Any:
        """Opens the while slide image file.

        Note: This shouldn't be called in the constructor as wsi backends usually contain
        C types or pointers, which the standard Python pickler cannot serialize, leading to
        issues with torch.DataLoader in multiprocessing settings.
        """
