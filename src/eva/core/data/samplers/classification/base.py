"""Base class for classification-based samplers."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Iterator, List, Union

import numpy as np
import torch
from typing_extensions import override

from eva.core.data import datasets
from eva.core.data.datasets.typings import DataSample
from eva.core.data.samplers.sampler import SamplerWithDataSource
from eva.core.utils.progress_bar import tqdm


class ClassificationSampler(SamplerWithDataSource[int], ABC):
    """Abstract base class for classification-based samplers.

    Provides common functionality for samplers that need to group samples by class
    and sample from each class according to some strategy.
    """

    def __init__(
        self,
        replacement: bool = False,
        seed: int | None = 42,
        reset_generator: bool = True,
    ) -> None:
        """Initializes the classification sampler.

        Args:
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            seed: Random seed for reproducibility.
            reset_generator: Whether to reset the random number generator
                when setting the dataset. This ensures that repeated runs that share the same
                sampler instance will start from the same seed.
        """
        self._replacement = replacement
        self._class_to_indices: Dict[Union[int, str], List[int]] = defaultdict(list)
        self._seed = seed
        self._reset_generator = reset_generator
        self._indices: List[int] = []
        self._random_generator: np.random.Generator

        self._set_generator()

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator that yields indices.

        Returns:
            Iterator yielding dataset indices.
        """
        return iter(self._indices)

    @override
    def set_dataset(self, data_source: datasets.MapDataset):
        """Sets the dataset and builds class indices.

        Args:
            data_source: The dataset to sample from.
        """
        super().set_dataset(data_source)
        if self._reset_generator:
            self._set_generator()
        self._build_class_to_indices()
        self._sample_indices()

    def _get_class(self, index: int) -> Union[int, str]:
        """Load and validate the class for a given sample index.

        Args:
            index: Index of the sample in the dataset.

        Returns:
            The class label (int or str) for the sample.

        Raises:
            ValueError: If target is None, not scalar, or unsupported type.
        """
        if hasattr(self.data_source, "load_target"):
            target = self.data_source.load_target(index)  # type: ignore
        else:
            _, target, _ = DataSample(*self.data_source[index])

        if target is None:
            raise ValueError("The dataset must return non-empty targets.")

        if isinstance(target, str) or isinstance(target, int):
            return target

        if isinstance(target, torch.Tensor):
            if target.numel() != 1:
                raise ValueError("The dataset must return a single & scalar target.")
            return int(target.item())

        raise ValueError("Unsupported target type. Expected str or tensor-like object.")

    def _build_class_to_indices(self) -> None:
        """Build a mapping from class to sample indices."""
        self._class_to_indices.clear()
        for idx in tqdm(range(len(self.data_source)), desc="Fetching class indices for sampler"):
            class_id = self._get_class(idx)
            self._class_to_indices[class_id].append(idx)

    @abstractmethod
    def _sample_indices(self) -> None:
        """Sample indices according to the sampling strategy.

        This method should populate self._indices with the sampled indices.
        Subclasses must implement this method to define their specific sampling strategy.
        """
        raise NotImplementedError

    def _set_generator(self) -> None:
        """Recreate the RNG so repeated runs reuse the same seed."""
        if self._seed is None:
            self._random_generator = np.random.default_rng()
            return
        self._random_generator = np.random.default_rng(self._seed)
