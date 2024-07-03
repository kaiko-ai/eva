"""Base Module for loading data from WSI files."""

import abc
from typing import Any, Sequence, Tuple

import numpy as np


class Wsi(abc.ABC):
    """Base class for loading data from Whole Slide Image (WSI) files."""

    def __init__(self, file_path: str, overwrite_mpp: float | None = None):
        """Initializes a Wsi object.

        Args:
            file_path: The path to the WSI file.
            overwrite_mpp: The microns per pixel (mpp) value to use when missing in WSI metadata.
        """
        self._wsi = self.open_file(file_path)
        self._overwrite_mpp = overwrite_mpp

    @abc.abstractmethod
    def open_file(self, file_path: str) -> Any:
        """Opens the WSI file.

        Args:
            file_path: The path to the WSI file.
        """

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
        """Microns per pixel at the highest resolution (level 0)."""

    @abc.abstractmethod
    def _read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Abstract method to read a region at a specified zoom level."""

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Reads and returns image data for a specified region and zoom level.

        Args:
            location: Top-left corner (x, y) to start reading at level 0.
            level: WSI level to read from.
            size: Region size as (width, height) in pixels at the selected read level.
                Remember to scale the size correctly.
        """
        self._verify_location(location, size)
        data = self._read_region(location, level, size)
        return self._read_postprocess(data)

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

    def _verify_location(self, location: Tuple[int, int], size: Tuple[int, int]) -> None:
        """Verifies that the requested region is within the slide dimensions.

        Args:
            location: Top-left corner (x, y) to start reading at level 0.
            size: Region size as (width, height) in pixels at the selected read level.
        """
        x_max, y_max = self.level_dimensions[0]
        x_scale = x_max / self.level_dimensions[0][0]
        y_scale = y_max / self.level_dimensions[0][1]

        if (
            int(location[0] + x_scale * size[0]) > x_max
            or int(location[1] + y_scale * size[1]) > y_max
        ):
            raise ValueError(f"Out of bounds region: {location}, {size}")

    def _read_postprocess(self, data: np.ndarray) -> np.ndarray:
        """Post-processes the read region data.

        Args:
            data: The read region data as a numpy array of shape (height, width, channels).
        """
        # Change color to white where the alpha channel is 0
        if data.shape[2] == 4:
            data[data[:, :, 3] == 0] = 255

        return data[:, :, :3]
