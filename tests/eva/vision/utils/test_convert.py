"""Tests image conversion related functionalities."""

from typing import Any

import numpy as np
import numpy.typing as npt
from pytest import mark

from eva.vision.utils import convert

IMAGE_ARRAY_INT16 = np.array(
    [
        [
            [-794, -339, -607, -950],
            [-608, -81, 172, -834],
            [-577, -10, 366, -837],
            [-790, -325, -564, -969],
        ]
    ],
    dtype=np.int16,
)
"""Test input data."""

EXPECTED_ARRAY_INT16 = np.array(
    [
        [
            [33, 120, 69, 3],
            [68, 169, 217, 25],
            [74, 183, 255, 25],
            [34, 123, 77, 0],
        ]
    ],
    dtype=np.uint8,
)
"""Test expected/desired features."""


@mark.parametrize(
    "image_array, expected",
    [
        [IMAGE_ARRAY_INT16, EXPECTED_ARRAY_INT16],
    ],
)
def test_to_8bit(
    image_array: npt.NDArray[Any],
    expected: npt.NDArray[np.uint8],
) -> None:
    """Tests the `to_8bit` image conversion."""
    actual = convert.to_8bit(image_array)
    np.testing.assert_allclose(actual, expected)
