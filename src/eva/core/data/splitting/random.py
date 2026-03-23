"""Functions for random splitting."""

from typing import Any, List, Sequence, Tuple

import numpy as np


def random_split(
    samples: Sequence[Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int] | None]:
    """Splits the samples into random train, validation, and test (optional) sets.

    Args:
        samples: The samples to split.
        train_ratio: The ratio of the training set.
        val_ratio: The ratio of the validation set.
        test_ratio: The ratio of the test set (optional).
        seed: The seed for reproducibility.

    Returns:
        The indices of the train, validation, and test sets as lists.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio > 1.0:
        raise ValueError("The sum of the ratios must be lower or equal to 1.")

    random_generator = np.random.default_rng(seed)
    n_samples = int(total_ratio * len(samples))
    indices = random_generator.permutation(len(samples))[:n_samples]

    n_train = int(np.floor(train_ratio * n_samples))
    n_val = n_samples - n_train if test_ratio == 0.0 else int(np.floor(val_ratio * n_samples)) or 1

    train_indices = list(indices[:n_train])
    val_indices = list(indices[n_train : n_train + n_val])
    test_indices = list(indices[n_train + n_val :]) if test_ratio > 0.0 else None

    return train_indices, val_indices, test_indices
