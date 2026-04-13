"""Tests for the stratified split function."""

from typing import List

import pytest

from eva.core.data import splitting


@pytest.mark.parametrize(
    "targets, train_ratio, val_ratio, test_ratio",
    [
        ([0] * 50 + [1] * 50, 0.8, 0.2, 0.0),
        ([0] * 50 + [1] * 50, 0.7, 0.15, 0.15),
        ([0] * 30 + [1] * 70, 0.8, 0.2, 0.0),
        ([0] * 30 + [1] * 70, 0.7, 0.15, 0.15),
        ([0] * 30 + [1] * 70, 0.2, 0.1, 0.15),
    ],
)
def test_stratification(
    targets: List[int], train_ratio: float, val_ratio: float, test_ratio: float
):
    """Tests if the stratified split maintains the class proportions."""
    samples = list(range(len(targets)))
    train_indices, val_indices, test_indices = splitting.stratified_split(
        samples, targets, train_ratio, val_ratio, test_ratio
    )

    train_classes = [targets[i] for i in train_indices]
    val_classes = [targets[i] for i in val_indices]

    for c in set(targets):
        expected_train_proportion = train_ratio * targets.count(c)
        expected_val_proportion = val_ratio * targets.count(c)
        assert train_classes.count(c) == pytest.approx(expected_train_proportion, abs=1)
        assert val_classes.count(c) == pytest.approx(expected_val_proportion, abs=1)

    if train_ratio + val_ratio + test_ratio == 1:
        assert len(train_indices) + len(val_indices) + len(test_indices or []) == len(samples)


@pytest.mark.parametrize("train_ratio, val_ratio, test_ratio", [(0.6, 0.5, 0.0), (0.6, 0.0, 0.7)])
def test_invalid_ratio_sums(train_ratio: float, val_ratio: float, test_ratio: float):
    """Tests if the function raises an error when the ratios do not sum to 1."""
    samples = list(range(100))
    targets = [0] * 50 + [1] * 50
    expected_error = "The sum of the ratios must be lower or equal to 1"
    with pytest.raises(ValueError, match=expected_error):
        splitting.stratified_split(samples, targets, train_ratio, val_ratio, test_ratio)


@pytest.mark.parametrize("seed1, seed2", [(42, 43), (123, 124), (999, 1000)])
def test_different_seeds_produce_different_outputs(seed1: int, seed2: int):
    """Tests if different seeds produce different train, validation, and test indices."""
    samples = list(range(100))
    targets = [0] * 50 + [1] * 50

    train1, val1, test1 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed1)
    train2, val2, test2 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed2)

    assert train1 != train2, "Different seeds should produce different train indices"
    assert val1 != val2, "Different seeds should produce different validation indices"
    assert test1 != test2, "Different seeds should produce different test indices"


@pytest.mark.parametrize(
    "seed, train_expected_indices, val_expected_indices, test_expected_indices",
    [
        (42, [5, 25, 20, 49], [3, 44, 30, 10], [0, 12, 14, 48]),
        (123, [15, 38, 41, 7], [29, 44, 46, 37], [16, 9, 12, 45]),
        (999, [49, 10, 1, 25], [24, 39, 3, 37], [0, 28, 13, 16]),
    ],
)
def test_same_seed_produces_same_outputs(
    seed: int,
    train_expected_indices: List[int],
    val_expected_indices: List[int],
    test_expected_indices: List[int],
):
    """Tests deterministic output for same seed."""
    samples = list(range(100))
    targets = [0] * 50 + [1] * 50

    train1, val1, test1 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed)
    train2, val2, test2 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed)

    assert train1 == train2, "Same seed should produce the same train indices"
    assert val1 == val2, "Same seed should produce the same validation indices"
    assert test1 == test2, "Same seed should produce the same test indices"
    assert isinstance(test1, list)

    assert train1[: len(train_expected_indices)] == train_expected_indices, "Unexpected indices"
    assert val1[: len(val_expected_indices)] == val_expected_indices, "Unexpected indices"
    assert test1[: len(test_expected_indices)] == test_expected_indices, "Unexpected indices"


def test_groups_length_mismatch():
    """Tests when the length of groups does not match the number of samples."""
    samples = list(range(5))
    targets = [0, 1, 0, 1, 0]
    groups = [0, 0, 0]  # wrong length

    with pytest.raises(ValueError):
        splitting.stratified_split(samples, targets, 0.6, 0.2, groups=groups)


def test_groups_are_not_split_across_sets():
    """Tests that samples from the same group are not split across split sets."""
    samples = list(range(10))
    targets = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    # two groups
    groups = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    train, val, test = splitting.stratified_split(
        samples, targets, 0.6, 0.2, 0.2, groups=groups, seed=42
    )

    for g in set(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]

        assigned = [
            any(i in train for i in idx),
            any(i in val for i in idx),
            any(i in test or [] for i in idx),
        ]

        # group must appear in exactly one split
        assert sum(assigned) == 1


def test_all_samples_still_assigned_with_groups():
    """Tests that all samples are assigned to a split when using groups."""
    samples = list(range(20))
    targets = [0] * 10 + [1] * 10

    groups = [i // 2 for i in range(20)]  # multiple small groups

    train, val, test = splitting.stratified_split(
        samples, targets, 0.7, 0.3, 0.0, groups=groups, seed=1
    )

    all_idx = set(train + val + (test or []))
    assert all_idx == set(samples)
