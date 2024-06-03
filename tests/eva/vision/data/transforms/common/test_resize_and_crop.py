"""Test the ResizeAndCrop augmentation."""

from typing import Tuple

import pytest
import torch
from torch import testing
from torchvision import tv_tensors

from eva.vision.data.transforms import common


@pytest.mark.parametrize(
    "image_size, target_size, expected_size, expected_mean",
    [
        ((3, 512, 224), [112, 224], (3, 112, 224), -0.00392),
        ((3, 224, 512), [112, 224], (3, 112, 224), -0.00392),
        ((3, 512, 224), [112, 97], (3, 112, 97), -0.00392),
        ((3, 512, 512), 224, (3, 224, 224), -0.00392),
        ((3, 512, 224), 224, (3, 224, 224), -0.00392),
        ((3, 224, 224), 224, (3, 224, 224), -0.00392),
        ((3, 97, 97), 224, (3, 224, 224), -0.00392),
    ],
)
def test_resize_and_crop(
    image_tensor: tv_tensors.Image,
    resize_and_crop: common.ResizeAndCrop,
    expected_size: Tuple[int, int, int],
    expected_mean: float,
) -> None:
    """Tests the ResizeAndCrop transform."""
    output = resize_and_crop(image_tensor)
    assert output.shape == expected_size
    testing.assert_close(output.mean(), torch.tensor(expected_mean))


@pytest.fixture(scope="function")
def resize_and_crop(target_size: Tuple[int, int, int]) -> common.ResizeAndCrop:
    """Transform ResizeAndCrop fixture."""
    return common.ResizeAndCrop(size=target_size)


@pytest.fixture(scope="function")
def image_tensor(image_size: Tuple[int, int, int]) -> tv_tensors.Image:
    """Image tensor fixture."""
    image_tensor = 127 * torch.ones(image_size, dtype=torch.uint8)
    return tv_tensors.Image(image_tensor)
