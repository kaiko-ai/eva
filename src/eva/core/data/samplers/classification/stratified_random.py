"""Stratified random sampler for classification tasks."""

from loguru import logger
from typing_extensions import override

from eva.core.data.samplers.classification.base import ClassificationSampler


class StratifiedRandomSampler(ClassificationSampler):
    """Stratified random sampler for data loading.

    The sampler ensures that:
    1. Class proportions from the original dataset are maintained
    2. Samples within each class are randomly selected
    3. Total number of samples matches the specified num_samples or sample_ratio
    """

    def __init__(
        self,
        num_samples: int | None = None,
        sample_ratio: float | None = None,
        replacement: bool = False,
        seed: int | None = 42,
        reset_generator: bool = True,
    ) -> None:
        """Initializes the stratified random sampler.

        Args:
            num_samples: The total number of samples to draw across all classes. If None, defaults
                to the dataset size.
            sample_ratio: Alternative to num_samples, specifies the dataset fraction to sample.
                If both num_samples and sample_ratio are provided, num_samples takes precedence.
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            seed: Random seed for reproducibility.
            reset_generator: Whether to reset the random number generator
                when setting the dataset. This ensures that repeated runs that share the same
                sampler instance will start from the same seed.
        """
        super().__init__(replacement=replacement, seed=seed, reset_generator=reset_generator)
        self._num_samples = num_samples
        self._sample_ratio = sample_ratio

    def __len__(self) -> int:
        """Returns the total number of samples."""
        dataset_size = len(self.data_source)
        if self._num_samples is not None:
            return self._num_samples
        if self._sample_ratio is not None:
            return int(round(dataset_size * self._sample_ratio))
        return dataset_size

    @override
    def _sample_indices(self) -> None:
        """Sample indices proportionally from each class to maintain class distribution."""
        total_dataset_samples = len(self.data_source)

        if self._num_samples is not None:
            total_samples_to_draw = self._num_samples
        elif self._sample_ratio is not None:
            total_samples_to_draw = int(round(total_dataset_samples * self._sample_ratio))
        else:
            self._indices = [idx for indices in self._class_to_indices.values() for idx in indices]
            self._random_generator.shuffle(self._indices)
            return

        self._indices = []
        samples_allocated = 0
        class_list = sorted(self._class_to_indices.keys())

        for i, class_id in enumerate(class_list):
            class_indices = self._class_to_indices[class_id]
            class_size = len(class_indices)

            if i == len(class_list) - 1:
                samples_for_class = total_samples_to_draw - samples_allocated
            else:
                class_proportion = class_size / total_dataset_samples
                samples_for_class = int(round(class_proportion * total_samples_to_draw))

            if not self._replacement and samples_for_class > class_size:
                logger.warning(
                    f"Class {class_id} requested {samples_for_class} samples but only has "
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
