"""Test the ResizeAndClamp augmentation."""

from typing import Tuple

import pytest
import torch
from torch import testing
from torchvision import tv_tensors

from eva.vision.data.transforms import common


@pytest.mark.parametrize(
    "image_size, target_size, clamp_range, expected_size, expected_mean",
    [
        ((3, 512, 224), [112, 224], [0, 255], (3, 112, 224), 0.498039186000824),
        ((3, 512, 224), [112, 224], [100, 155], (3, 112, 224), 0.4909091293811798),
        ((3, 224, 512), [112, 224], [0, 100], (3, 112, 224), 1.0),
        ((3, 512, 224), [112, 97], [100, 255], (3, 112, 97), 0.17419354617595673),
        ((3, 512, 512), 224, [0, 255], (3, 224, 224), 0.4980391561985016),
    ],
)
def test_resize_and_clamp(
    image_tensor: tv_tensors.Image,
    resize_and_clamp: common.ResizeAndClamp,
    expected_size: Tuple[int, int, int],
    expected_mean: float,
) -> None:
    """Tests the ResizeAndClamp transform."""
    output = resize_and_clamp(image_tensor)
    assert output.shape == expected_size
    testing.assert_close(torch.tensor(expected_mean), output.mean())


@pytest.fixture(scope="function")
def resize_and_clamp(
    target_size: Tuple[int, int, int], clamp_range: Tuple[int, int]
) -> common.ResizeAndClamp:
    """Transform ResizeAndClamp fixture."""
    return common.ResizeAndClamp(size=target_size, clamp_range=clamp_range)


@pytest.fixture(scope="function")
def image_tensor(image_size: Tuple[int, int, int]) -> tv_tensors.Image:
    """Image tensor fixture."""
    image_tensor = 127 * torch.ones(image_size, dtype=torch.uint8)
    return tv_tensors.wrap(image_tensor, like=image_tensor)  # type: ignore
