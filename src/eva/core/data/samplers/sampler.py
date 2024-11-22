"""Core data sampler."""

from typing import TypeVar, Generic
from torch.utils import data
from eva.core.data import datasets

Sampler = data.Sampler
"""Core abstract data sampler class."""

T_co = TypeVar('T_co', covariant=True)

class SamplerWithDataSource(Sampler, Generic[T_co]):
    """A sampler base class that enables to specify the data source after initialization."""

    data_source: datasets.MapDataset

    def set_dataset(self, data_source: datasets.MapDataset) -> None:
        """Sets the dataset to sample from.

        Args:
            data_source: The dataset to sample from.
        """
        self.data_source = data_source
