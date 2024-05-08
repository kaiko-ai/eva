"""GridSampler tests."""

from typing import Tuple

import pytest

from eva.vision.data.wsi.patching import samplers

TEST_ARGS = {"width": 10, "height": 10, "layer_shape": (100, 100)}


@pytest.mark.parametrize("max_samples, expected_n_samples", [(3, 3), (10, 10), (200, 100)])
def test_length(max_samples: int, expected_n_samples: int) -> None:
    """Tests if the sampler returns the correct number of samples."""
    sampler = samplers.GridSampler(max_samples=max_samples)

    x_y = list(sampler.sample(**TEST_ARGS))

    assert len(x_y) == expected_n_samples


@pytest.mark.parametrize("max_samples, seed", [(10, 8), (22, 42)])
def test_same_seed(max_samples: int, seed: int) -> None:
    """Tests if the sampler returns the same samples for the same seed."""
    sampler = samplers.GridSampler(max_samples=max_samples, seed=seed)

    x_y_1 = list(sampler.sample(**TEST_ARGS))
    x_y_2 = list(sampler.sample(**TEST_ARGS))

    assert x_y_1 == x_y_2


@pytest.mark.parametrize("max_samples, seed_1, seed_2", [(3, 1, 2), (5, 3, 4)])
def test_different_seed(max_samples: int, seed_1: int, seed_2: int) -> None:
    """Tests if the sampler returns different samples for different seeds."""
    sampler_1 = samplers.GridSampler(max_samples=max_samples, seed=seed_1)
    sampler_2 = samplers.GridSampler(max_samples=max_samples, seed=seed_2)

    x_y_1 = list(sampler_1.sample(**TEST_ARGS))
    x_y_2 = list(sampler_2.sample(**TEST_ARGS))

    assert x_y_1 != x_y_2


def test_invalid_width_height() -> None:
    """Tests if the sampler raises an error when width / height is bigger than layer_shape."""
    sampler = samplers.GridSampler(max_samples=10, seed=42)

    with pytest.raises(ValueError):
        list(sampler.sample(width=200, height=200, layer_shape=(100, 100)))


@pytest.mark.parametrize(
    "width, height, layer_shape",
    [
        (5, 5, (25, 25)),
        (5, 5, (100, 100)),
        (224, 224, (1000, 1000)),
    ],
)
def test_expected_n_patches(width: int, height: int, layer_shape: Tuple[int, int]) -> None:
    """Tests if the sampler respects the max_samples limit."""
    sampler = samplers.GridSampler(max_samples=None)

    expected_max_samples = (layer_shape[0] // width) * (layer_shape[1] // height)

    x_y = list(sampler.sample(width=width, height=height, layer_shape=layer_shape))

    assert len(x_y) == expected_max_samples
