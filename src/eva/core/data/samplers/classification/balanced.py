"""Random class sampler for data loading."""

from collections import defaultdict
from typing import Dict, Iterator, List

import numpy as np
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

    def __init__(self, num_samples: int, replacement: bool = False, seed: int | None = 42):
        """Initializes the balanced sampler.

        Args:
            num_samples: The number of samples to draw per class.
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            seed: Random seed for reproducibility.
        """
        self._num_samples = num_samples
        self._replacement = replacement
        self._class_indices: Dict[int, List[int]] = defaultdict(list)
        self._random_generator = np.random.default_rng(seed)
        self._indices: List[int] = []

    def __len__(self) -> int:
        """Returns the total number of samples."""
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
        self._make_indices()

    def _make_indices(self):
        """Samples the indices for each class in the dataset."""
        self._class_indices.clear()
        for idx in tqdm(range(len(self.data_source)), desc="Fetching class indices for sampler"):
            if hasattr(self.data_source, "load_target"):
                target = self.data_source.load_target(idx)  # type: ignore
            else:
                _, target, _ = DataSample(*self.data_source[idx])
            if target is None:
                raise ValueError("The dataset must return non-empty targets.")
            if target.numel() != 1:
                raise ValueError("The dataset must return a single & scalar target.")

            class_idx = int(target.item())
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
