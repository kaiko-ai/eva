"""RandomSampler tests."""

import pytest

from eva.vision.data.wsi.patching import samplers

TEST_ARGS = {"width": 10, "height": 10, "layer_shape": (100, 100)}


@pytest.mark.parametrize("n_samples", [3, 10, 22])
def test_length(n_samples: int) -> None:
    """Tests if the sampler returns the correct number of samples."""
    sampler = samplers.RandomSampler(n_samples=n_samples)

    x_y = list(sampler.sample(**TEST_ARGS))

    assert len(x_y) == n_samples


@pytest.mark.parametrize("n_samples, seed", [(10, 8), (22, 42)])
def test_same_seed(n_samples: int, seed: int) -> None:
    """Tests if the sampler returns the same samples for the same seed."""
    sampler = samplers.RandomSampler(n_samples=n_samples, seed=seed)

    x_y_1 = list(sampler.sample(**TEST_ARGS))
    x_y_2 = list(sampler.sample(**TEST_ARGS))

    assert x_y_1 == x_y_2


@pytest.mark.parametrize("n_samples, seed_1, seed_2", [(10, 1, 2), (22, 3, 4)])
def test_different_seed(n_samples: int, seed_1: int, seed_2: int) -> None:
    """Tests if the sampler returns different samples for different seeds."""
    sampler_1 = samplers.RandomSampler(n_samples=n_samples, seed=seed_1)
    sampler_2 = samplers.RandomSampler(n_samples=n_samples, seed=seed_2)

    x_y_1 = list(sampler_1.sample(**TEST_ARGS))
    x_y_2 = list(sampler_2.sample(**TEST_ARGS))

    assert x_y_1 != x_y_2


def test_invalid_width_height() -> None:
    """Tests if the sampler raises an error when width / height is bigger than layer_shape."""
    sampler = samplers.RandomSampler(n_samples=10, seed=42)

    with pytest.raises(ValueError):
        list(sampler.sample(width=200, height=200, layer_shape=(100, 100)))
