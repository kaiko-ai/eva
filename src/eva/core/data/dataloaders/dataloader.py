"""Core Dataloader module."""

import dataclasses
import multiprocessing
from typing import Callable

from torch.utils.data import dataloader

from eva.core.data import datasets, samplers


@dataclasses.dataclass
class DataLoader:
    """The `DataLoader` combines a dataset and a sampler.

    It provides an iterable over the given dataset.
    """

    batch_size: int | None = 1
    """How many samples per batch to load.

    Set to `None` for iterable dataset where dataset produces batches.
    """

    shuffle: bool = False
    """Whether to shuffle the data at every epoch."""

    sampler: samplers.Sampler | None = None
    """Defines the strategy to draw samples from the dataset.

    Can be any Iterable with `__len__` implemented. If specified, shuffle must
    not be specified.
    """

    batch_sampler: samplers.Sampler | None = None
    """Like `sampler`, but returns a batch of indices at a time.

    Mutually exclusive with `batch_size`, `shuffle`, `sampler` and `drop_last`.
    """

    num_workers: int = multiprocessing.cpu_count()
    """How many workers to use for loading the data.

    By default, it will use the number of CPUs available.
    """

    collate_fn: Callable | None = None
    """The batching process."""

    pin_memory: bool = True
    """Will copy Tensors into CUDA pinned memory before returning them."""

    drop_last: bool = False
    """Drops the last incomplete batch."""

    persistent_workers: bool = True
    """Will keep the worker processes after a dataset has been consumed once."""

    prefetch_factor: int | None = 2
    """Number of batches loaded in advance by each worker."""

    def __call__(self, dataset: datasets.TorchDataset) -> dataloader.DataLoader:
        """Returns the dataloader on the provided dataset.

        Args:
            dataset: dataset from which to load the data.
        """
        return dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
