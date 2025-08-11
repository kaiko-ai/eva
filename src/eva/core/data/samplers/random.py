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
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the random sampler.

        Args:
            replacement: Samples are drawn on-demand with replacement if ``True``, default=``False``.
            num_samples: Number of samples to draw, default=``len(dataset)``.
            generator: Optional torch.Generator used for sampling.
            seed: Optional seed for the random number generator.
        """
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator or torch.Generator()
        self.seed = seed

        if self.seed is not None:
            self.generator.manual_seed(self.seed)

    @override
    def set_dataset(self, data_source: datasets.MapDataset) -> None:
        super().__init__(
            data_source,
            replacement=self.replacement,
            num_samples=self.num_samples,
            generator=self.generator,
        )