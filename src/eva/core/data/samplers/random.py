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
        sample_ratio: Optional[float] = None,
        seed: Optional[int] = None,
        reset_generator: bool = True,
    ) -> None:
        """Initialize the random sampler.

        Args:
            replacement: Samples are drawn on-demand with replacement if ``True``, default=``False``
            num_samples: Number of samples to draw, default=``len(dataset)``.
            sample_ratio: Alternative to num_samples, specifies the dataset fraction to sample.
                If both num_samples and sample_ratio are provided, num_samples takes precedence.
            seed: Optional seed for the random number generator.
            reset_generator: Whether to reset the random number generator
                when setting the dataset. This ensures that repeated runs that share the same
                sampler instance will start from the same seed.
        """
        self.replacement = replacement
        self._num_samples = num_samples
        self._sample_ratio = sample_ratio
        self._seed = seed
        self._reset_generator = reset_generator
        self._generator = None

        self._set_generator()

    @override
    def set_dataset(self, data_source: datasets.MapDataset) -> None:
        if self._reset_generator:
            self._set_generator()

        num_samples = self._num_samples
        if num_samples is None and self._sample_ratio is not None:
            dataset_size = len(data_source)
            num_samples = int(round(dataset_size * self._sample_ratio))
        if num_samples is None and self._sample_ratio is None:
            num_samples = len(data_source)

        super().__init__(
            data_source,
            replacement=self.replacement,
            num_samples=num_samples,
            generator=self._generator,
        )

    def _set_generator(self) -> None:
        if self._seed is None:
            self._generator = None
        else:
            self._generator = self._generator or torch.Generator()
            self._generator.manual_seed(self._seed)
