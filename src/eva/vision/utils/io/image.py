"""Image I/O related functions."""

import cv2
import numpy as np
import numpy.typing as npt
from typing import Any

from eva.vision.utils.io import _utils


def read_image(path: str) -> npt.NDArray[np.uint8]:
    """Reads and loads the image from a file path as a RGB.

    Args:
        path: The path of the image file.

    Returns:
        The RGB image as a numpy array.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        IOError: If the image could not be loaded.
    """
    return read_image_as_array(path, cv2.IMREAD_COLOR)


def read_image_as_array(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> npt.NDArray[np.uint8]:
    """Reads and loads an image file as a numpy array.

    Args:
        path: The path to the image file.
        flags: Specifies the way in which the image should be read.

    Returns:
        The image as a numpy array.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        IOError: If the image could not be loaded.
    """
    _utils.check_file(path)
    image = cv2.imread(path, flags=flags)
    if image is None:
        raise IOError(
            f"Input '{path}' could not be loaded. "
            "Please verify that the path is a valid image file."
        )

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.ndim == 2 and flags == cv2.IMREAD_COLOR:
        image = image[:, :, np.newaxis]

    return np.asarray(image).astype(np.uint8)


def get_mask(
    wsi: Any,
    wsi_path: str,
    level_idx: int,
    kernel_size: tuple[int, int] = (7, 7),
    gray_threshold: int = 220,
    fill_holes: bool = False,
) -> tuple[np.ndarray, float]:
    """Extracts a binary mask from an image.
    
    Args:
        image: The input image.
        kernel_size: The size of the kernel for morphological operations.
        gray_threshold: The threshold for the gray scale image.
        fill_holes: Whether to fill holes in the mask.
    """
    image = np.array(
        wsi.open_file(wsi_path).read_region(
            [0, 0], len(wsi.level_dimensions) - 1, wsi.level_dimensions[-1]
        )
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)

    if fill_holes:
        mask = cv2.dilate(mask, kernel, iterations=1)
        contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask, [cnt], 0, 1, -1)

    mask_scale_factor = wsi.level_dimensions[-1][0] / wsi.level_dimensions[level_idx][0]

    return mask, mask_scale_factor
