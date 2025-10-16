"""Random class sampler for data loading."""

from collections import defaultdict
from typing import Dict, Iterator, List, Union

import numpy as np
import torch
from loguru import logger
from typing_extensions import override

from eva.core.data import datasets
from eva.core.data.datasets.typings import DataSample
from eva.core.data.samplers.sampler import SamplerWithDataSource
from eva.core.utils.progress_bar import tqdm


class BalancedSampler(SamplerWithDataSource[int]):
    """Balanced class sampler for data loading.

    The sampler ensures that:
    1. Each class has the same number of samples
    2. Samples within each class are randomly selected
    3. Samples of different classes appear in random order
    """

    def __init__(
        self,
        num_samples: int | None,
        replacement: bool = False,
        seed: int | None = 42,
        reset_generator: bool = True,
    ) -> None:
        """Initializes the balanced sampler.

        Args:
            num_samples: The number of samples to draw per class.
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            seed: Random seed for reproducibility.
            reset_generator: Whether to reset the random number generator
                when setting the dataset. This ensures that repeated runs that share the same
                sampler instance will start from the same seed.
        """
        self._num_samples = num_samples
        self._replacement = replacement
        self._class_indices: Dict[Union[int, str], List[int]] = defaultdict(list)
        self._seed = seed
        self._reset_generator = reset_generator
        self._indices: List[int] = []
        self._random_generator: np.random.Generator

        self._set_generator()

    def __len__(self) -> int:
        """Returns the total number of samples."""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples * len(self._class_indices)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator that yields indices in a class balanced way.

        Returns:
            Iterator yielding dataset indices.
        """
        return iter(self._indices)

    @override
    def set_dataset(self, data_source: datasets.MapDataset):
        """Sets the dataset and builds class indices.

        Args:
            data_source: The dataset to sample from.

        Raises:
            ValueError: If the dataset doesn't have targets or if any class has
                fewer samples than `num_samples` and `replacement` is `False`.
        """
        super().set_dataset(data_source)
        if self._reset_generator:
            self._set_generator()
        self._make_indices()

    def _get_class_idx(self, idx):
        """Load and validate the class index for a given sample index."""
        if hasattr(self.data_source, "load_target"):
            target = self.data_source.load_target(idx)  # type: ignore
        else:
            _, target, _ = DataSample(*self.data_source[idx])

        if target is None:
            raise ValueError("The dataset must return non-empty targets.")

        if isinstance(target, str) or isinstance(target, int):
            return target

        if isinstance(target, torch.Tensor):
            if target.numel() != 1:
                raise ValueError("The dataset must return a single & scalar target.")
            return int(target.item())

        raise ValueError("Unsupported target type. Expected str or tensor-like object.")

    def _make_indices(self):
        """Samples the indices for each class in the dataset."""
        if self._num_samples is None:
            self._indices = list(self._random_generator.permutation(len(self.data_source)))
            return
        self._class_indices.clear()
        for idx in tqdm(range(len(self.data_source)), desc="Fetching class indices for sampler"):
            class_idx = self._get_class_idx(idx)
            self._class_indices[class_idx].append(idx)

        if not self._replacement:
            for class_idx, indices in self._class_indices.items():
                if len(indices) < self._num_samples:
                    raise ValueError(
                        f"Class {class_idx} has only {len(indices)} samples, "
                        f"which is less than the required {self._num_samples} samples."
                    )

        self._indices = []
        for class_idx in self._class_indices:
            class_indices = self._class_indices[class_idx]
            sampled_indices = self._random_generator.choice(
                class_indices, size=self._num_samples, replace=self._replacement
            ).tolist()
            self._indices.extend(sampled_indices)
        self._random_generator.shuffle(self._indices)
        logger.debug(f"Sampled indices: {self._indices}")

    def _set_generator(self) -> None:
        """Recreate the RNG so repeated runs reuse the same seed."""
        if self._seed is None:
            self._random_generator = np.random.default_rng()
            return
        self._random_generator = np.random.default_rng(self._seed)
