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
