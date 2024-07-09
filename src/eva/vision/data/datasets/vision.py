"""Vision Dataset base class."""

import abc
from typing import Generic, TypeVar

from eva.core.data.datasets import base

DataSample = TypeVar("DataSample")
"""The data sample type."""


class VisionDataset(base.Dataset, abc.ABC, Generic[DataSample]):
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

    @abc.abstractmethod
    def __getitem__(self, index: int) -> DataSample:
        """Returns the `index`'th data sample.

        Args:
            index: The index of the data-sample to select.

        Returns:
            A data sample and its target.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the total length of the data."""
        raise NotImplementedError
