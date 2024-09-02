"""Test the rescale intensity transform."""

from typing import Tuple

import pytest
import torch
from torch import testing
from torchvision import tv_tensors

from eva.vision.data.transforms.normalization import functional


@pytest.mark.parametrize(
    "image_size, in_range, out_range, expected_mean",
    [
        ((3, 224, 224), (0, 255), (0.0, 1.0), 0.4980391561985016),
        ((3, 224, 224), (0, 255), (-0.5, 0.5), -0.001960783964022994),
        ((3, 224, 224), (100, 155), (-0.5, 0.5), -0.009090900421142578),
        ((3, 224, 224), (100, 155), (0.0, 1.0), 0.4909091293811798),
    ],
)
def test_rescale_intensity(
    image_tensor: tv_tensors.Image,
    in_range: Tuple[int, int],
    out_range: Tuple[int, int],
    expected_mean: float,
) -> None:
    """Tests the rescale_intensity functional transform."""
    output = functional.rescale_intensity(image_tensor, in_range=in_range, out_range=out_range)
    testing.assert_close(torch.tensor(expected_mean), output.mean())


@pytest.fixture(scope="function")
def image_tensor(image_size: Tuple[int, int, int]) -> tv_tensors.Image:
    """Image tensor fixture."""
    image_tensor = 127 * torch.ones(image_size, dtype=torch.uint8)
    return tv_tensors.wrap(image_tensor, like=image_tensor)  # type: ignore
