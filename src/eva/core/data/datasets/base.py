"""Base dataset class."""

import abc
from typing import Generic, TypeVar

from eva.core.data.datasets import dataset


class Dataset(dataset.TorchDataset):
    """Base dataset class."""

    def prepare_data(self) -> None:
        """Encapsulates all disk related tasks.

        This method is preferred for downloading and preparing the data, for
        example generate manifest files. If implemented, it will be called via
        :class:`eva.core.data.datamodules.DataModule`, which ensures that is called
        only within a single process, making it multi-processes safe.
        """

    def setup(self) -> None:
        """Setups the dataset.

        This method is preferred for creating datasets or performing
        train/val/test splits. If implemented, it will be called via
        :class:`eva.core.data.datamodules.DataModule` at the beginning of fit
        (train + validate), validate, test, or predict and it will be called
        from every process (i.e. GPU) across all the nodes in DDP.
        """
        self.configure()
        self.validate()

    def configure(self):
        """Configures the dataset.

        This method is preferred to configure the dataset; assign values
        to attributes, perform splits etc. This would be called from the
        method ::method::`setup`, before calling the ::method::`validate`.
        """

    def validate(self):
        """Validates the dataset.

        This method aims to check the integrity of the dataset and verify
        that is configured properly. This would be called from the method
        ::method::`setup`, after calling the ::method::`configure`.
        """

    def teardown(self) -> None:
        """Cleans up the data artifacts.

        Used to clean-up when the run is finished. If implemented, it will
        be called via :class:`eva.core.data.datamodules.DataModule` at the end
        of fit (train + validate), validate, test, or predict and it will be
        called from every process (i.e. GPU) across all the nodes in DDP.
        """


DataSample = TypeVar("DataSample")
"""The data sample type."""


class MapDataset(Dataset, abc.ABC, Generic[DataSample]):
    """Abstract base class for all map-style datasets."""

    @abc.abstractmethod
    def __getitem__(self, index: int) -> DataSample:
        """Retrieves the item at the given index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            The data at the given index.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        raise NotImplementedError
