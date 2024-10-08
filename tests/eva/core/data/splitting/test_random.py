"""Tests for the random split function."""

from typing import List

import pytest

from eva.core.data import splitting


@pytest.mark.parametrize(
    "n_samples, train_ratio, val_ratio, test_ratio",
    [
        (100, 0.8, 0.2, 0.0),
        (100, 0.7, 0.15, 0.15),
        (150, 0.8, 0.2, 0.0),
        (150, 0.7, 0.15, 0.15),
    ],
)
def test_split_ratios(n_samples: int, train_ratio: float, val_ratio: float, test_ratio: float):
    """Tests if the random split maintains the correct ratios."""
    samples = list(range(n_samples))
    train_indices, val_indices, test_indices = splitting.random_split(
        samples, train_ratio, val_ratio, test_ratio
    )

    assert len(train_indices) == pytest.approx(n_samples * train_ratio, abs=1)
    assert len(val_indices) == pytest.approx(n_samples * val_ratio, abs=1)
    if test_ratio > 0:
        assert isinstance(test_indices, list)
        assert len(test_indices) == pytest.approx(n_samples * test_ratio, abs=1)
    else:
        assert test_indices is None

    assert len(train_indices) + len(val_indices) + len(test_indices or []) == n_samples


@pytest.mark.parametrize("train_ratio, val_ratio, test_ratio", [(0.6, 0.7, 0.0), (0.6, 0.4, 0.3)])
def test_invalid_ratio_sums(train_ratio: float, val_ratio: float, test_ratio: float):
    """Tests if the function raises an error when the ratios do not sum to 1."""
    samples = list(range(100))
    expected_error = "The sum of the ratios must be lower or equal to 1"
    with pytest.raises(ValueError, match=expected_error):
        splitting.random_split(samples, train_ratio, val_ratio, test_ratio)


@pytest.mark.parametrize("seed1, seed2", [(42, 43), (123, 124), (999, 1000)])
def test_different_seeds_produce_different_outputs(seed1, seed2):
    """Tests if different seeds produce different train, validation, and test indices."""
    samples = list(range(100))
    train1, val1, test1 = splitting.random_split(samples, 0.6, 0.2, 0.2, seed=seed1)
    train2, val2, test2 = splitting.random_split(samples, 0.6, 0.2, 0.2, seed=seed2)

    assert train1 != train2, "Different seeds should produce different train indices"
    assert val1 != val2, "Different seeds should produce different validation indices"
    assert test1 != test2, "Different seeds should produce different test indices"


@pytest.mark.parametrize(
    "seed, train_expected_indices, val_expected_indices, test_expected_indices",
    [
        (42, [59, 21, 56, 18], [69, 15, 48, 55], [49, 6, 90, 11]),
        (123, [21, 71, 92, 23], [89, 14, 64, 4], [45, 75, 62, 6]),
        (999, [47, 42, 57, 50], [41, 3, 81, 61], [45, 6, 56, 67]),
    ],
)
def test_same_seed_produces_same_outputs(
    seed: int,
    train_expected_indices: List[int],
    val_expected_indices: List[int],
    test_expected_indices: List[int],
):
    """Tests if the same seed produces the same train, validation, and test indices."""
    samples = list(range(100))
    train1, val1, test1 = splitting.random_split(samples, 0.6, 0.2, 0.2, seed=seed)
    train2, val2, test2 = splitting.random_split(samples, 0.6, 0.2, 0.2, seed=seed)

    assert train1 == train2, "Same seed should produce the same train indices"
    assert val1 == val2, "Same seed should produce the same validation indices"
    assert test1 == test2, "Same seed should produce the same test indices"
    assert isinstance(test1, list)

    assert train1[: len(train_expected_indices)] == train_expected_indices, "Unexpected indices"
    assert val1[: len(val_expected_indices)] == val_expected_indices, "Unexpected indices"
    assert test1[: len(test_expected_indices)] == test_expected_indices, "Unexpected indices"


def test_no_test_set():
    """Tests if the function correctly handles the case when test_ratio is 0."""
    samples = list(range(100))
    train_indices, val_indices, test_indices = splitting.random_split(samples, 0.8, 0.2, 0.0)

    assert len(train_indices) + len(val_indices) == len(samples)
    assert test_indices is None


def test_all_samples_used():
    """Tests if all samples are used in the split."""
    samples = list(range(100))
    train_indices, val_indices, test_indices = splitting.random_split(samples, 0.6, 0.2, 0.2)

    assert isinstance(test_indices, list)
    all_indices = set(train_indices + val_indices + test_indices)
    assert all_indices == set(samples)


def test_no_overlap():
    """Tests if there's no overlap between train, validation, and test sets."""
    samples = list(range(100))
    train_indices, val_indices, test_indices = splitting.random_split(samples, 0.6, 0.2, 0.2)

    assert isinstance(test_indices, list)
    assert len(set(train_indices) & set(val_indices)) == 0
    assert len(set(train_indices) & set(test_indices)) == 0
    assert len(set(val_indices) & set(test_indices)) == 0
