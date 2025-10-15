"""Stratified random sampler for classification tasks."""

import numpy as np
from loguru import logger
from typing_extensions import override

from eva.core.data.samplers.classification.base import ClassificationSampler


class StratifiedRandomSampler(ClassificationSampler):
    """Stratified random sampler for data loading.

    The sampler ensures that:
    1. Class proportions from the original dataset are maintained
    2. Samples within each class are randomly selected
    3. Total number of samples matches the specified num_samples
    """

    def __init__(
        self,
        num_samples: int | None = None,
        replacement: bool = False,
        seed: int | None = 42,
        reset_generator: bool = True,
    ) -> None:
        """Initializes the stratified random sampler.

        Args:
            num_samples: The total number of samples to draw across all classes. If None, defaults
                to the dataset size.
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            seed: Random seed for reproducibility.
            reset_generator: Whether to reset the random number generator
                when setting the dataset. This ensures that repeated runs that share the same
                sampler instance will start from the same seed.
        """
        super().__init__(replacement=replacement, seed=seed, reset_generator=reset_generator)
        self._num_samples = num_samples

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return self._num_samples or len(self.data_source)

    @override
    def _sample_indices(self) -> None:
        """Sample indices proportionally from each class to maintain class distribution."""
        total_dataset_samples = len(self.data_source)

        if not self._num_samples:
            self._num_samples = (
                total_dataset_samples  # Set here as data_source is not available in __init__
            )
            self._replacement = False  # No replacement if sampling entire dataset

        self._indices = []
        samples_allocated = 0
        class_list = sorted(self._class_indices.keys())

        for i, class_idx in enumerate(class_list):
            class_indices = self._class_indices[class_idx]
            class_size = len(class_indices)

            # Last class gets remaining samples to meet total, otherwise proportionally rounded
            if i == len(class_list) - 1:
                samples_for_class = self._num_samples - samples_allocated
            else:
                class_proportion = class_size / total_dataset_samples
                samples_for_class = int(np.round(class_proportion * self._num_samples))

            if not self._replacement:
                if samples_for_class > class_size:
                    logger.warning(
                        f"Class {class_idx} needs {samples_for_class} samples but only has "
                        f"{class_size}. Using all available samples."
                    )
                    samples_for_class = class_size

            if samples_for_class > 0:
                sampled_indices = self._random_generator.choice(
                    class_indices, size=samples_for_class, replace=self._replacement
                ).tolist()
                self._indices.extend(sampled_indices)
                samples_allocated += samples_for_class

        self._random_generator.shuffle(self._indices)

        logger.debug(f"Sampled {len(self._indices)} indices maintaining class proportions")
