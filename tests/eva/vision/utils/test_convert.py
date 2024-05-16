"""Tests image conversion related functionalities."""

import torch
from pytest import mark

from eva.vision.utils import convert

IMAGE_ONE = torch.Tensor(
    [
        [
            [-794, -339, -607, -950],
            [-608, -81, 172, -834],
            [-577, -10, 366, -837],
            [-790, -325, -564, -969],
        ]
    ],
).to(dtype=torch.float16)
"""Test input data."""

EXPECTED_ONE = torch.Tensor(
    [
        [
            [33, 120, 69, 3],
            [69, 169, 217, 25],
            [74, 183, 255, 25],
            [34, 123, 77, 0],
        ]
    ],
).to(dtype=torch.uint8)
"""Test expected/desired features."""


@mark.parametrize(
    "image, expected",
    [
        [IMAGE_ONE, EXPECTED_ONE],
    ],
)
def test_descale_and_denorm_image(
    image: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Tests the `descale_and_denorm_image` image conversion."""
    actual = convert.descale_and_denorm_image(image, mean=(0.0,), std=(1.0,))
    torch.testing.assert_close(actual, expected)
