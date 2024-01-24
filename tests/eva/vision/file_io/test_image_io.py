"""Tests for the image_io module."""
import os

import numpy as np
import pytest

from eva.vision.file_io import load_image_file, load_image_file_as_rgb


@pytest.fixture()
def image_path_gray(assets_path: str) -> str:
    """Path to a grayscale test image."""
    return os.path.join(assets_path, "images", "random_grayscale_32x32.png")


@pytest.fixture()
def image_path_bgr(assets_path: str) -> str:
    """Path to a bgr test image."""
    return os.path.join(assets_path, "images", "random_bgr_32x32.png")


def test_load_image_file(image_path_gray: str, image_path_bgr: str) -> None:
    """Test the load_image_file function to load grayscale and bgr images."""
    image = load_image_file(image_path_gray)
    assert isinstance(image, np.ndarray), "The function should return a numpy array"
    assert image.dtype == np.uint8, "Data type of the array should be np.uint8"
    assert image.shape == (32, 32), "Shape of the array should be (32, 32)"

    image = load_image_file(image_path_bgr)
    assert isinstance(image, np.ndarray), "The function should return a numpy array"
    assert image.dtype == np.uint8, "Data type of the array should be np.uint8"
    assert image.shape == (32, 32, 3), "Shape of the array should be (32, 32, 3)"


def test_load_grayscale_image_file_as_rgb(image_path_gray: str) -> None:
    """Test to load and convert a grayscale image to RGB format."""
    image = load_image_file_as_rgb(image_path_gray)
    assert image.dtype == np.uint8, "Data type of the array should be np.uint8"
    assert image.shape == (32, 32, 3), "Shape of the array should be (32, 32, 3)"
