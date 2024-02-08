"""Tests for the image IO functions."""

import os

import numpy as np
import pytest

from eva.vision.utils import io


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        ["random_bgr_32x32.png", (32, 32, 3)],
        ["random_grayscale_32x32.png", (32, 32, 3)],
    ],
)
def test_read_image(
    image_path: str,
    expected_shape: tuple,
) -> None:
    """Test the read_image function."""
    image = io.read_image(image_path)
    assert isinstance(image, np.ndarray), f"Invalid type {type(image)} != numpy.ndarray."
    assert image.dtype == np.uint8, f"Invalid numpy data type {image.dtype} != np.uint8."
    assert image.shape == expected_shape, f"Invalid shape {image.shape} != {expected_shape}."


@pytest.fixture
def image_path(filename: str, assets_path: str) -> str:
    """Returns the full path to the image file."""
    return os.path.join(assets_path, "images", filename)
