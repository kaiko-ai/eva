"""Test the ResizeAndNormalize augmentation."""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import testing

from eva.vision.data.transforms import common


@pytest.mark.parametrize(
    "image_size, target_size, expected_size, expected_mean",
    [
        ((512, 512, 3), 224, (3, 224, 224), -0.00392),
        ((512, 224, 3), 224, (3, 224, 224), -0.00392),
        ((224, 224, 3), 224, (3, 224, 224), -0.00392),
        ((97, 97, 3), 224, (3, 224, 224), -0.00392),
    ],
)
def test_resize_and_normalize(
    image_array: npt.NDArray,
    resize_and_normalize: common.ResizeAndNormalize,
    expected_size: Tuple[int, int, int],
    expected_mean: float,
) -> None:
    """Tests the ResizeAndNormalize transform."""
    output = resize_and_normalize(image_array)
    assert output.shape == expected_size
    testing.assert_close(output.mean(), torch.tensor(expected_mean))


@pytest.fixture(scope="function")
def resize_and_normalize(target_size: Tuple[int, int, int]) -> common.ResizeAndNormalize:
    """Transform ResizeAndNormalize fixture."""
    return common.ResizeAndNormalize(size=target_size)


@pytest.fixture(scope="function")
def image_array(image_size: Tuple[int, int, int]) -> npt.NDArray:
    """Image array fixture."""
    return 127 * np.ones(image_size, np.uint8)
