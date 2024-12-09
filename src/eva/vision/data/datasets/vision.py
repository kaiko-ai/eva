"""Vision Dataset base class."""

import abc
from typing import Generic, TypeVar

from eva.core.data.datasets import base

DataSample = TypeVar("DataSample")
"""The data sample type."""


class VisionDataset(base.MapDataset, abc.ABC, Generic[DataSample]):
    """Base dataset class for vision tasks."""

    @abc.abstractmethod
    def filename(self, index: int) -> str:
        """Returns the filename of the `index`'th data sample.

        Note that this is the relative file path to the root.

        Args:
            index: The index of the data-sample to select.

        Returns:
            The filename of the `index`'th data sample.
        """
