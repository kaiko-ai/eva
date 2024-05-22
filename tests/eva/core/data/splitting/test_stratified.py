"""Tests for the stratified split function."""

import pytest

from eva.core.data import splitting


@pytest.mark.parametrize(
    "targets, train_ratio, val_ratio, test_ratio",
    [
        ([0] * 50 + [1] * 50, 0.8, 0.2, 0.0),
        ([0] * 50 + [1] * 50, 0.7, 0.15, 0.15),
        ([0] * 30 + [1] * 70, 0.8, 0.2, 0.0),
        ([0] * 30 + [1] * 70, 0.7, 0.15, 0.15),
    ],
)
def test_stratification(
    targets: list[int], train_ratio: float, val_ratio: float, test_ratio: float
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

    assert len(train_indices) + len(val_indices) + len(test_indices or []) == len(samples)


@pytest.mark.parametrize("train_ratio, val_ratio, test_ratio", [(0.6, 0.3, 0.0), (0.6, 0.4, 0.3)])
def test_invalid_ratio_sums(train_ratio: float, val_ratio: float, test_ratio: float):
    """Tests if the function raises an error when the ratios do not sum to 1."""
    samples = list(range(100))
    targets = [0] * 50 + [1] * 50
    expected_error = "The sum of the ratios must be equal to 1."
    with pytest.raises(ValueError, match=expected_error):
        splitting.stratified_split(samples, targets, train_ratio, val_ratio, test_ratio)


@pytest.mark.parametrize("seed1, seed2", [(42, 43), (123, 124), (999, 1000)])
def test_different_seeds_produce_different_outputs(seed1, seed2):
    """Tests if different seeds produce different train, validation, and test indices."""
    samples = list(range(100))
    targets = [0] * 50 + [1] * 50
    train1, val1, test1 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed1)
    train2, val2, test2 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed2)

    assert train1 != train2, "Different seeds should produce different train indices"
    assert val1 != val2, "Different seeds should produce different validation indices"
    assert test1 != test2, "Different seeds should produce different test indices"


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_same_seed_produces_same_outputs(seed):
    """Tests if the same seed produces the same train, validation, and test indices."""
    samples = list(range(100))
    targets = [0] * 50 + [1] * 50
    train1, val1, test1 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed)
    train2, val2, test2 = splitting.stratified_split(samples, targets, 0.6, 0.2, 0.2, seed=seed)

    assert train1 == train2, "Same seed should produce the same train indices"
    assert val1 == val2, "Same seed should produce the same validation indices"
    assert test1 == test2, "Same seed should produce the same test indices"
