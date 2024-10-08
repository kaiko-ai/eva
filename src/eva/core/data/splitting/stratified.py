"""Functions for stratified splitting."""

from typing import Any, List, Sequence, Tuple

import numpy as np


def stratified_split(
    samples: Sequence[Any],
    targets: Sequence[Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int] | None]:
    """Splits the samples into stratified train, validation, and test (optional) sets.

    Args:
        samples: The samples to split.
        targets: The corresponding targets used for stratification.
        train_ratio: The ratio of the training set.
        val_ratio: The ratio of the validation set.
        test_ratio: The ratio of the test set (optional).
        seed: The seed for reproducibility.

    Returns:
        The indices of the train, validation, and test sets.
    """
    if len(samples) != len(targets):
        raise ValueError("The number of samples and targets must be equal.")
    if train_ratio + val_ratio + (test_ratio or 0) > 1.0:
        raise ValueError("The sum of the ratios must be lower or equal to 1.")

    use_all_samples = train_ratio + val_ratio + test_ratio == 1
    random_generator = np.random.default_rng(seed)
    unique_classes, y_indices = np.unique(targets, return_inverse=True)
    n_classes = unique_classes.shape[0]

    train_indices, val_indices, test_indices = [], [], []

    for c in range(n_classes):
        class_indices = np.where(y_indices == c)[0]
        random_generator.shuffle(class_indices)

        n_train = int(np.floor(train_ratio * len(class_indices))) or 1
        n_val = (
            len(class_indices) - n_train
            if test_ratio == 0.0 and use_all_samples
            else int(np.floor(val_ratio * len(class_indices))) or 1
        )

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        if test_ratio > 0.0:
            n_test = (
                len(class_indices) - n_train - n_val
                if use_all_samples
                else int(np.floor(test_ratio * len(class_indices))) or 1
            )
            test_indices.extend(class_indices[n_train + n_val : n_train + n_val + n_test])

    return train_indices, val_indices, test_indices or None
