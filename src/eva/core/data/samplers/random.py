
from eva.core.data.samplers.sampler import SamplerWithDataSource
from eva.core.data import datasets
from typing import Optional
from torch.utils import data
from typing_extensions import override


class RandomSampler(data.RandomSampler, SamplerWithDataSource[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: datasets.MapDataset  # type: ignore
    replacement: bool

    def __init__(self, replacement: bool = False, num_samples: Optional[int] = None, generator=None) -> None:
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @override
    def set_dataset(self, data_source: datasets.MapDataset) -> None:
        super().__init__(data_source, replacement=self.replacement, num_samples=self.num_samples, generator=self.generator)
