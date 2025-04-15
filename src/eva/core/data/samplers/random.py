"""Random sampler for data loading."""

from typing import Optional

from torch.utils import data
from typing_extensions import override

from eva.core.data import datasets
from eva.core.data.samplers.sampler import SamplerWithDataSource
import torch


class RandomSampler(data.RandomSampler, SamplerWithDataSource[int]):
    """Samples elements randomly."""

    data_source: datasets.MapDataset  # type: ignore

    def __init__(
        self, replacement: bool = False, num_samples: Optional[int] = None, seed: int = 42
    ) -> None:
        """Initializes the random sampler.

        Args:
            data_source: dataset to sample from
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            num_samples: number of samples to draw, default=`len(dataset)`.
        seed: Random seed for reproducibility.
        """
        self._replacement = replacement
        self._num_samples = num_samples
        self._generator = self._get_generator(seed)

    @override
    def set_dataset(self, data_source: datasets.MapDataset) -> None:
        super().__init__(
            data_source,
            replacement=self._replacement,
            num_samples=self._num_samples,
            generator=self._generator,
        )

    def _get_generator(self, seed: int) -> torch.Generator:
        """Returns the randbom generator used for sampling."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
