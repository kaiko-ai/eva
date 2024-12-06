"""Core data sampler."""

from typing import Generic, TypeVar

from torch.utils import data

from eva.core.data import datasets

Sampler = data.Sampler
"""Core abstract data sampler class."""

T_co = TypeVar("T_co", covariant=True)


class SamplerWithDataSource(Sampler, Generic[T_co]):
    """A sampler base class that enables to specify the data source after initialization.

    The `set_dataset` can also be overwritten to expand the functionality of the derived
    sampler classes.
    """

    data_source: datasets.MapDataset

    def set_dataset(self, data_source: datasets.MapDataset) -> None:
        """Sets the dataset to sample from.

        This is not done in the constructor because the dataset might not be
        available at that time.

        Args:
            data_source: The dataset to sample from.
        """
        self.data_source = data_source
