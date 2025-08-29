"""Tests for image encoding utilities."""

import base64

import pytest
import torch
from torchvision import tv_tensors

from eva.multimodal.utils.image.encode import encode_image


def test_encode_image_base64():
    """Test base64 encoding of image tensors."""
    image = tv_tensors.Image(torch.rand(3, 224, 224))
    encoded = encode_image(image, encoding="base64")

    assert isinstance(encoded, str)
    # Test that it's valid base64
    base64.b64decode(encoded)
    assert len(encoded) > 0


def test_encode_image_unsupported_encoding():
    """Test that unsupported encoding raises ValueError."""
    image = tv_tensors.Image(torch.rand(3, 224, 224))

    with pytest.raises(ValueError, match="Unsupported encoding type"):
        encode_image(image, encoding="unsupported")  # type: ignore


@pytest.mark.parametrize("image_shape", [(3, 32, 32), (3, 224, 224), (3, 512, 512)])
def test_encode_different_sizes(image_shape):
    """Test encoding works with different image sizes."""
    image = tv_tensors.Image(torch.rand(*image_shape))
    encoded = encode_image(image, encoding="base64")

    assert isinstance(encoded, str)
    assert len(encoded) > 0
