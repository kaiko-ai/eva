""""Balanced class sampler for data loading."""

from loguru import logger
from typing_extensions import override

from eva.core.data.samplers.classification.base import ClassificationSampler


class BalancedSampler(ClassificationSampler):
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
            num_samples: The number of samples to draw per class. if None, sampling will
                be disabled and the whole dataset will be used (shuffled).
            replacement: samples are drawn on-demand with replacement if ``True``, default=``False``
            seed: Random seed for reproducibility.
            reset_generator: Whether to reset the random number generator
                when setting the dataset. This ensures that repeated runs that share the same
                sampler instance will start from the same seed.
        """
        self._num_samples = num_samples
        super().__init__(replacement=replacement, seed=seed, reset_generator=reset_generator)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples * len(self._class_to_indices)

    @override
    def _sample_indices(self) -> None:
        """Sample equal number of indices from each class."""
        if self._num_samples is None:
            self._indices = list(self._random_generator.permutation(len(self.data_source)))
            return
        if not self._replacement:
            for class_id, indices in self._class_to_indices.items():
                if len(indices) < self._num_samples:
                    raise ValueError(
                        f"Class {class_id} has only {len(indices)} samples, "
                        f"which is less than the required {self._num_samples} samples."
                    )

        self._indices = []
        for class_id in self._class_to_indices:
            class_indices = self._class_to_indices[class_id]
            sampled_indices = self._random_generator.choice(
                class_indices, size=self._num_samples, replace=self._replacement
            ).tolist()
            self._indices.extend(sampled_indices)

        self._random_generator.shuffle(self._indices)
        logger.debug(
            f"Sampled {len(self._indices)} indices from {len(self._class_to_indices)} classes"
        )
