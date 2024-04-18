"""Base Module for loading data from WSI files."""

import abc
from typing import Any, Sequence, Tuple

import numpy as np


class Wsi(abc.ABC):
    """Base class for loading data from WSI (whole slide image) files."""

    def __init__(self, file_path: str):
        """Initializes a new class instance.

        Args:
            file_path: The path to the whole slide image file.
        """
        self._file_path = file_path
        self._wsi = None

    @property
    @abc.abstractmethod
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        """A list of (width, height) tuples for each level, from highest to lowest resolution."""

    @property
    @abc.abstractmethod
    def level_downsamples(self) -> Sequence[float]:
        """A list of downsampling factors for each level, relative to the highest resolution."""

    @property
    @abc.abstractmethod
    def mpp(self) -> float:
        """Microns per pixel at the highest resolution."""

    @abc.abstractmethod
    def read_region(
        self, location: Tuple[int, int], size: Tuple[int, int], level: int
    ) -> np.ndarray:
        """Reads and returns image data for a specified region and zoom level.

        Args:
            location: Top-left corner (x, y) to start reading.
            size: Region size as (width, height), relative to <location>.
            level: Zoom level, with 0 being the highest resolution.
        """

    @abc.abstractmethod
    def open_slide(self) -> Any:
        """Opens the WSI file.

        Note: This shouldn't be called in the constructor as wsi backends usually contain
        C types or pointers, which the standard Python pickler cannot serialize, leading to
        issues with torch.DataLoader in multiprocessing settings.
        """

    def get_closest_level(self, target_mpp: float) -> int:
        """Calculate the slide level that is closest to the target mpp.

        Args:
            slide: The whole-slide image object.
            target_mpp: The target microns per pixel (mpp) value.
        """
        # Calculate the mpp for each level
        level_mpps = self.mpp * np.array(self.level_downsamples)

        # Ignore levels with higher mpp
        level_mpps_filtered = level_mpps.copy()
        level_mpps_filtered[level_mpps_filtered > target_mpp] = 0

        if level_mpps_filtered.max() == 0:
            # When all levels have higher mpp than target_mpp return the level with lowest mpp
            level_idx = np.argmin(level_mpps)
        else:
            level_idx = np.argmax(level_mpps_filtered)

        return int(level_idx)
