"""ForegroundGridSampler tests."""

import numpy as np
import pytest

from eva.vision.data.wsi.patching import mask, samplers

TEST_MASK = mask.Mask(
    mask_array=np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    ),
    mask_level_idx=3,
    scale_factors=(6.0, 6.0),
)

TEST_ARGS = {"width": 12, "height": 12, "layer_shape": (36, 36), "mask": TEST_MASK}


@pytest.mark.parametrize(
    "min_foreground_ratio, max_samples, expected_n_samples",
    [(0.0, 3, 3), (0.0, 100, 9), (0.5, 100, 5), (0.9, 100, 1)],
)
def test_length(min_foreground_ratio: float, max_samples: int, expected_n_samples: int) -> None:
    """Tests if the sampler returns the correct number of samples."""
    sampler = samplers.ForegroundGridSampler(
        max_samples=max_samples, min_foreground_ratio=min_foreground_ratio
    )

    x_y = list(sampler.sample(**TEST_ARGS))

    assert len(x_y) == expected_n_samples


@pytest.mark.parametrize(
    "max_samples, seed, x_y_expected",
    [
        (7, 42, [(12, 0), (24, 12), (12, 12), (0, 12), (12, 24)]),
        (10, 8, [(12, 0), (12, 24), (24, 12), (0, 12), (12, 12)]),
    ],
)
def test_same_seed(max_samples: int, seed: int, x_y_expected: list) -> None:
    """Tests if the sampler returns the same samples for the same seed."""
    sampler = samplers.ForegroundGridSampler(
        max_samples=max_samples, seed=seed, min_foreground_ratio=0.5
    )

    x_y_1 = list(sampler.sample(**TEST_ARGS))
    x_y_2 = list(sampler.sample(**TEST_ARGS))

    assert x_y_1 == x_y_2
    assert x_y_1 == x_y_expected


@pytest.mark.parametrize("max_samples, seed_1, seed_2", [(3, 1, 2), (5, 3, 4)])
def test_different_seed(max_samples: int, seed_1: int, seed_2: int) -> None:
    """Tests if the sampler returns different samples for different seeds."""
    sampler_1 = samplers.ForegroundGridSampler(max_samples=max_samples, seed=seed_1)
    sampler_2 = samplers.ForegroundGridSampler(max_samples=max_samples, seed=seed_2)

    x_y_1 = list(sampler_1.sample(**TEST_ARGS))
    x_y_2 = list(sampler_2.sample(**TEST_ARGS))

    assert x_y_1 != x_y_2


def test_invalid_width_height() -> None:
    """Tests if the sampler raises an error when width / height is bigger than layer_shape."""
    sampler = samplers.ForegroundGridSampler(max_samples=10, seed=42)

    with pytest.raises(ValueError):
        list(sampler.sample(width=200, height=200, layer_shape=(100, 100), mask=TEST_MASK))


@pytest.mark.parametrize("min_foreground_ratio", [0.0, 0.5, 0.9])
def test_min_foreground_ratio(min_foreground_ratio: float) -> None:
    """Tests if sampled coordinates respect the min_foreground_ratio."""
    sampler = samplers.ForegroundGridSampler(
        max_samples=100, min_foreground_ratio=min_foreground_ratio
    )

    x_y = list(sampler.sample(**TEST_ARGS))

    mask = TEST_MASK
    width, height = TEST_ARGS["width"], TEST_ARGS["height"]

    for x, y in x_y:
        x_, y_ = sampler._scale_coords(x, y, mask.scale_factors)
        width_, height_ = sampler._scale_coords(width, height, mask.scale_factors)

        patch_mask = mask.mask_array[x_ : x_ + width_, y_ : y_ + height_]
        foreground_ratio = patch_mask.sum() / patch_mask.size

        assert foreground_ratio >= min_foreground_ratio
