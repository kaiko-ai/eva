"""Core Dataset module."""
import abc
from typing import Generic, TypeVar

from eva.data.datasets.dataset import Dataset

DataSample = TypeVar("DataSample")
"""The data sample type."""


class VisionDataset(Dataset, abc.ABC, Generic[DataSample]):
    """Base dataset class for vision tasks."""

    def prepare_data(self) -> None:
        """Correspons to the `prepare_data` method from LightningDataModule.

        Lightning ensures the prepare_data() is called only within a single process on CPU and there
        is a barrier in between which ensures that all the processes proceed to setup(). So this is
        the place where you can do things like:

            - download the dataset
            - generate manifest files
            - ...
        """

    def setup(self) -> None:
        """Correspons to the `setup` method from LightningDataModule.

        The setup method is called after prepare_data() and on every worker. Use setup() to do
        things like:

            - perform dataset splits
            - count number of classes
            - ...
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
