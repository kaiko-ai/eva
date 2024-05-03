"""Functions for extracting foreground masks."""

import dataclasses
from typing import Tuple

import cv2
import numpy as np

from eva.vision.data.wsi.backends.base import Wsi


@dataclasses.dataclass
class Mask:
    """A class to store the mask of a whole-slide image."""

    mask_array: np.ndarray
    """Binary mask array where 1s represent the foreground and 0s represent the background."""

    scale_factor: float
    """Factor to scale mask to the wsi coordinates."""


def get_mask(
    wsi: Wsi,
    level_idx: int,
    kernel_size: Tuple[int, int] = (7, 7),
    gray_threshold: int = 220,
    fill_holes: bool = False,
) -> Mask:
    """Extracts a binary mask from an image.

    Args:
        wsi: The WSI object.
        level_idx: The level index of the WSI at which we specify the coordinates.
        kernel_size: The size of the kernel for morphological operations.
        gray_threshold: The threshold for the gray scale image.
        fill_holes: Whether to fill holes in the mask.
    """
    low_res_level = get_lowest_resolution_level(wsi, min_pixels=1000 * 1000)
    image = wsi.read_region((0, 0), low_res_level, wsi.level_dimensions[low_res_level])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    gray = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)

    if fill_holes:
        mask = cv2.dilate(mask, kernel, iterations=1)
        contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask, [cnt], 0, (1,), -1)

    mask_scale_factor = wsi.level_dimensions[low_res_level][0] / wsi.level_dimensions[level_idx][0]

    return Mask(mask_array=mask, scale_factor=mask_scale_factor)


def get_lowest_resolution_level(wsi: Wsi, min_pixels: int | None):
    """Calculates the WSI level corresponding to the lowest resolution/magnification.

    Args:
        wsi: The WSI object.
        min_pixels: If specified, this funciton will return the lowest resolution
            level with an area of at least `min_pixels` pixels.

    Returns:
        The lowest resolution level index of the given WSI.
    """
    valid_level_index = len(wsi.level_dimensions) - 1

    if min_pixels is None:
        return valid_level_index
    else:
        for index, (width, height) in reversed(list(enumerate(wsi.level_dimensions))):
            if width * height >= min_pixels:
                valid_level_index = index
                break

    return valid_level_index
