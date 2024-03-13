"""Vision Dataset base class."""

import abc
from typing import Generic, TypeVar

from eva.data.datasets.dataset import Dataset

DataSample = TypeVar("DataSample")
"""The data sample type."""


class VisionDataset(Dataset, abc.ABC, Generic[DataSample]):
    """Base dataset class for vision tasks."""

    def prepare_data(self) -> None:
        """Encapsulates all disk related tasks.

        This method is preferred for downloading and preparing the data, for
        example generate manifest files. If implemented, it will be called via
        :class:`eva.data.datamodules.DataModule`, which ensures that is called
        only within a single process, making it multi-processes safe.
        """

    def setup(self) -> None:
        """Setups the dataset.

        This method is preferred for creating datasets or performing
        train/val/test splits. If implemented, it will be called via
        :class:`eva.data.datamodules.DataModule` at the beginning of fit
        (train + validate), validate, test, or predict and it will be called
        from every process (i.e. GPU) across all the nodes in DDP.
        """

    def teardown(self) -> None:
        """Cleans up the data artifacts.

        Used to clean-up when the run is finished. If implemented, it will
        be called via :class:`eva.data.datamodules.DataModule` at the end
        of fit (train + validate), validate, test, or predict and it will be
        called from every process (i.e. GPU) across all the nodes in DDP.
        """

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
