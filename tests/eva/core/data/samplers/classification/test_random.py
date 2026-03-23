"""Tests for the random sampler."""

import pytest

from eva.core.data.samplers import RandomSampler
from tests.eva.core.data.samplers import _utils


@pytest.mark.parametrize("reset_generator", [True, False])
def test_reset_generator_on_multiple_runs(reset_generator: bool):
    """Sampler should produce identical samples when reused with the same seed."""
    dataset = _utils.multiclass_dataset(num_samples=30, num_classes=3)
    sampler = RandomSampler(num_samples=5, seed=123, reset_generator=reset_generator)
    sampler.set_dataset(dataset)
    first_run = list(sampler)
    sampler.set_dataset(dataset)
    second_run = list(sampler)

    if reset_generator:
        assert first_run == second_run
    else:
        assert first_run != second_run


def test_sample_ratio():
    """Test sampling with sample_ratio parameter."""
    dataset = _utils.multiclass_dataset(num_samples=100, num_classes=3)
    sampler = RandomSampler(sample_ratio=0.3, seed=42)
    sampler.set_dataset(dataset)
    samples = list(sampler)

    assert len(samples) == 30


def test_sample_ratio_with_rounding():
    """Test that sample_ratio rounds correctly."""
    dataset = _utils.multiclass_dataset(num_samples=27, num_classes=3)
    sampler = RandomSampler(sample_ratio=0.5, seed=42)
    sampler.set_dataset(dataset)
    samples = list(sampler)

    assert len(samples) == 14  # 27 * 0.5 = 13.5


def test_num_samples_takes_precedence_over_ratio():
    """Test that num_samples takes precedence when both are provided."""
    dataset = _utils.multiclass_dataset(num_samples=100, num_classes=3)
    sampler = RandomSampler(num_samples=10, sample_ratio=0.5, seed=42)
    sampler.set_dataset(dataset)
    samples = list(sampler)

    assert len(samples) == 10  # num_samples


def test_no_params_uses_full_dataset():
    """Test that not providing num_samples or sample_ratio uses the full dataset."""
    dataset = _utils.multiclass_dataset(num_samples=48, num_classes=3)
    sampler = RandomSampler(seed=42)
    sampler.set_dataset(dataset)
    samples = list(sampler)

    assert len(samples) == 48
