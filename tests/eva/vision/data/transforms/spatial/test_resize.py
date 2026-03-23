"""Tests for resize transforms."""

import pytest
import torch
from torchvision import tv_tensors

from eva.vision.data import transforms
from eva.vision.utils.image import encode as encode_utils


@pytest.mark.parametrize(
    "size, input_shape, expected_shape",
    [
        # Square images with tuple size
        ((100, 100), (3, 200, 200), (3, 100, 100)),
        ((50, 50), (3, 500, 500), (3, 50, 50)),
        # Non-square images with tuple size
        ((100, 200), (3, 400, 400), (3, 100, 200)),
        ((200, 100), (3, 300, 300), (3, 200, 100)),
        ((128, 256), (3, 512, 512), (3, 128, 256)),
        # Non-square input images with tuple size
        ((100, 100), (3, 200, 400), (3, 100, 100)),
        ((150, 200), (3, 300, 600), (3, 150, 200)),
        # Int size (smaller edge becomes this size, aspect ratio preserved)
        (100, (3, 200, 200), (3, 100, 100)),
        (100, (3, 200, 400), (3, 100, 200)),
        (100, (3, 400, 200), (3, 200, 100)),
        (256, (3, 512, 1024), (3, 256, 512)),
        (256, (3, 1024, 512), (3, 512, 256)),
    ],
)
def test_resize_with_size_only(size, input_shape, expected_shape):
    """Test Resize with only size parameter provided."""
    resize_transform = transforms.Resize(size=size)
    test_image = tv_tensors.Image(torch.rand(*input_shape))

    result = resize_transform(test_image)

    assert isinstance(result, tv_tensors.Image)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "size, max_size, input_shape, expected_max_dimension",
    [
        (100, 150, (3, 200, 200), 150),
        (100, 120, (3, 400, 800), 120),
        (200, 300, (3, 1000, 500), 300),
        (150, 256, (3, 512, 512), 256),
        (200, 300, (3, 600, 800), 300),
    ],
)
def test_resize_with_max_size(size, max_size, input_shape, expected_max_dimension):
    """Test Resize with max_size parameter to constrain the longer edge.

    When size is an int, it specifies the smaller edge size, and max_size
    constrains the longer edge. max_size must be strictly greater than size.
    """
    resize_transform = transforms.Resize(size=size, max_size=max_size)
    test_image = tv_tensors.Image(torch.rand(*input_shape))

    result = resize_transform(test_image)

    assert isinstance(result, tv_tensors.Image)
    # Check that the longer edge doesn't exceed max_size
    assert max(result.shape[1], result.shape[2]) <= expected_max_dimension


@pytest.mark.parametrize(
    "max_bytes, input_shape",
    [
        (1000, (3, 500, 500)),  # Square image
        (1000, (3, 600, 400)),  # Non-square image (landscape)
        (1000, (3, 400, 600)),  # Non-square image (portrait)
        (1500, (3, 1000, 1000)),  # Large image with low byte limit
    ],
)
def test_resize_with_max_bytes_only(max_bytes, input_shape):
    """Test Resize with only max_bytes parameter provided."""
    resize_transform = transforms.Resize(max_bytes=max_bytes)
    test_image = tv_tensors.Image(torch.rand(*input_shape))

    num_bytes_before = len(encode_utils.encode_image(test_image))
    assert num_bytes_before > max_bytes

    result = resize_transform(test_image)

    num_bytes_after = len(encode_utils.encode_image(result))
    assert num_bytes_after <= max_bytes
    assert isinstance(result, tv_tensors.Image)
    assert result.shape[1] < input_shape[1] or result.shape[2] < input_shape[2]


@pytest.mark.parametrize(
    "input_shape",
    [
        (3, 200, 200),  # Square image
        (3, 300, 400),  # Non-square image (landscape)
        (3, 400, 300),  # Non-square image (portrait)
        (3, 512, 512),
        (1, 100, 100),  # Single channel
        (4, 256, 256),  # 4 channels (e.g., RGBA)
    ],
)
def test_resize_with_no_parameters(input_shape):
    """Test Resize with neither size nor max_bytes parameters provided."""
    resize_transform = transforms.Resize()
    test_image = tv_tensors.Image(torch.rand(*input_shape))

    result = resize_transform(test_image)

    assert isinstance(result, tv_tensors.Image)
    # Should return original image unchanged
    assert result.shape == test_image.shape
    assert torch.equal(result, test_image)


@pytest.mark.parametrize(
    "invalid_max_bytes",
    [0, -1, -100, -1000],
)
def test_resize_with_invalid_max_bytes(invalid_max_bytes):
    """Test Resize raises ValueError for non-positive max_bytes."""
    with pytest.raises(ValueError, match="'max_bytes' must be a positive integer."):
        transforms.Resize(max_bytes=invalid_max_bytes)


@pytest.mark.parametrize(
    "input_shape",
    [
        (3, 100, 100),  # Square image
        (3, 200, 300),  # Non-square image (landscape)
        (3, 300, 200),  # Non-square image (portrait)
    ],
)
def test_resize_preserves_input_type(input_shape):
    """Test that Resize preserves the input tensor type (Image/Mask)."""
    resize_transform = transforms.Resize(size=(50, 50))

    # Test with Image
    test_image = tv_tensors.Image(torch.rand(*input_shape))
    result_image = resize_transform(test_image)
    assert isinstance(result_image, tv_tensors.Image)

    # Test with Mask
    test_mask = tv_tensors.Mask(torch.randint(0, 10, input_shape))
    result_mask = resize_transform(test_mask)
    assert isinstance(result_mask, tv_tensors.Mask)
