import abc
from typing import Any


class Wsi(abc.ABC):
    def __init__(self, file_path: str):
        """Initializes a new class instance.

        Args:
            file_path: The path to the whole slide image file.
        """
        self._file_path = file_path

        self._slide = self.open_slide(file_path)

    @abc.abstractmethod
    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        """A list of (width, height) tuples for each zoom level, from highest to lowest resolution."""

    @abc.abstractmethod
    @property
    def level_downsamples(self) -> list[float]:
        """A list of downsampling factors for each zoom level, relative to the highest resolution."""

    @abc.abstractmethod
    @property
    def mpp(self) -> float:
        """Microns per pixel at the highest resolution, indicating physical size per image pixel."""

    @abc.abstractmethod
    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> any:
        """Reads and returns image data for a specified region and zoom level.

        Args:
            location: Top-left corner (x, y) to start reading.
            level: Zoom level, with 0 being the highest resolution.
            size: Region size as (width, height).
        """

    @abc.abstractmethod
    @staticmethod
    def open_slide(file_path: str) -> Any:
        """Opens a while slide image file.

        Args:
            file_path: Path to the whole slide image file.
        """
