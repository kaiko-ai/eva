"""Functions for stratified splitting."""

from typing import Any, List, Sequence, Tuple, Iterable

import numpy as np


def stratified_split(
    samples: Sequence[Any] | Iterable[Any],
    targets: Sequence[Any] | Iterable[Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float = 0.0,
    groups: Sequence[Any] | Iterable[Any] | None = None,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int] | None]:
    """Splits the samples into stratified train, validation, and test (optional) sets.

    Args:
        samples: The samples to split.
        targets: The corresponding targets used for stratification.
        train_ratio: The ratio of the training set.
        val_ratio: The ratio of the validation set.
        test_ratio: The ratio of the test set (optional).
        groups: Optional group labels for group-wise stratification.
        seed: The seed for reproducibility.

    Returns:
        The indices of the train, validation, and test sets.
    """
    samples_seq = samples if isinstance(samples, (list, tuple)) else list(samples)
    targets_seq = targets if isinstance(targets, (list, tuple)) else list(targets)
    
    if len(samples_seq) != len(targets_seq):
        raise ValueError("The number of samples and targets must be equal.")
    
    if groups is not None:
        groups_seq = groups if isinstance(groups, (list, tuple)) else list(groups)
        if len(groups_seq) != len(samples_seq):
            raise ValueError("The number of samples and groups must be equal.")
        
        unique_groups, group_indices = np.unique(groups_seq, return_inverse=True)
        group_targets = np.array(
            [targets_seq[np.where(group_indices == i)[0][0]] for i in range(len(unique_groups))]
        )

        train_g, val_g, test_g = stratified_split(
            samples=unique_groups.tolist(),
            targets=group_targets.tolist(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        def map_indices(g_list):
            if g_list is None:
                return []
            selected_groups = unique_groups[g_list]
            return np.where(np.isin(groups_seq, selected_groups))[0].tolist()

        return map_indices(train_g), map_indices(val_g), map_indices(test_g) or None

    use_all_samples = train_ratio + val_ratio + test_ratio == 1
    random_generator = np.random.default_rng(seed)
    unique_classes, y_indices = np.unique(targets_seq, return_inverse=True)
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
