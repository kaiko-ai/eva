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


@pytest.mark.parametrize(
    "n_samples, seed, x_y_expected",
    [
        (7, 42, [(81, 14), (3, 35), (31, 28), (17, 13), (86, 69), (11, 75), (54, 4)]),
        (
            10,
            8,
            [
                (29, 47),
                (48, 16),
                (24, 90),
                (5, 10),
                (17, 31),
                (64, 26),
                (51, 82),
                (3, 58),
                (62, 58),
                (49, 63),
            ],
        ),
    ],
)
def test_same_seed(n_samples: int, seed: int, x_y_expected: int) -> None:
    """Tests if the sampler returns the same samples for the same seed."""
    sampler_1 = samplers.RandomSampler(n_samples=n_samples, seed=seed)
    sampler_2 = samplers.RandomSampler(n_samples=n_samples, seed=seed)

    x_y_1 = list(sampler_1.sample(**TEST_ARGS))
    x_y_2 = list(sampler_2.sample(**TEST_ARGS))

    assert x_y_1 == x_y_2
    assert x_y_1 == x_y_expected


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
