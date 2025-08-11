"""Random sampler for data loading."""

from typing import Optional

import torch
from torch.utils import data
from typing_extensions import override

from eva.core.data import datasets
from eva.core.data.samplers.sampler import SamplerWithDataSource


class RandomSampler(data.RandomSampler, SamplerWithDataSource[int]):
    """Samples elements randomly from a MapDataset."""

    data_source: datasets.MapDataset  # type: ignore

    def __init__(
        self,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the random sampler.

        Args:
            replacement: Samples are drawn on-demand with replacement if ``True``, default=``False``
            num_samples: Number of samples to draw, default=``len(dataset)``.
            seed: Optional seed for the random number generator.
        """
        self.replacement = replacement
        self._num_samples = num_samples
        self._generator = None

        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    @override
    def set_dataset(self, data_source: datasets.MapDataset) -> None:
        super().__init__(
            data_source,
            replacement=self.replacement,
            num_samples=self._num_samples,
            generator=self._generator,
        )
