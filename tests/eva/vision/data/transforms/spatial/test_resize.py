"""Tests for resize transforms."""

import pytest
import torch
from torchvision import tv_tensors

from eva.vision.data import transforms
from eva.vision.utils.image import encode as encode_utils


def test_resize_with_size_only():
    """Test Resize with only size parameter provided."""
    resize_transform = transforms.Resize(size=(100, 100))
    test_image = tv_tensors.Image(torch.rand(3, 200, 200))

    result = resize_transform(test_image)

    assert isinstance(result, tv_tensors.Image)
    assert result.shape == (3, 100, 100)


def test_resize_with_max_bytes_only():
    """Test Resize with only max_bytes parameter provided."""
    resize_transform = transforms.Resize(max_bytes=1000)  # Very small size to force resizing
    test_image = tv_tensors.Image(torch.rand(3, 500, 500))

    num_bytes_before = len(encode_utils.encode_image(test_image))
    assert num_bytes_before > 1000

    result = resize_transform(test_image)

    num_bytes_after = len(encode_utils.encode_image(result))
    assert num_bytes_after <= 1000
    assert isinstance(result, tv_tensors.Image)
    assert result.shape[1] < 500 or result.shape[2] < 500


def test_resize_with_no_parameters():
    """Test Resize with neither size nor max_bytes parameters provided."""
    resize_transform = transforms.Resize()
    test_image = tv_tensors.Image(torch.rand(3, 200, 200))

    result = resize_transform(test_image)

    assert isinstance(result, tv_tensors.Image)
    # Should return original image unchanged
    assert result.shape == test_image.shape
    assert torch.equal(result, test_image)


def test_resize_with_invalid_max_bytes():
    """Test Resize raises ValueError for non-positive max_bytes."""
    with pytest.raises(ValueError, match="'max_bytes' must be a positive integer."):
        transforms.Resize(max_bytes=0)

    with pytest.raises(ValueError, match="'max_bytes' must be a positive integer."):
        transforms.Resize(max_bytes=-100)
