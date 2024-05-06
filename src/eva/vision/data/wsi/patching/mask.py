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

    mask_level_idx: int
    """WSI level index at which the mask_array was extracted."""

    scale_factors: Tuple[float, float]
    """Factors to scale x/y coordinates from mask_level_idx to level 0."""


def get_mask(
    wsi: Wsi,
    mask_level_idx: int,
    kernel_size: Tuple[int, int] = (7, 7),
    gray_threshold: int = 220,
    fill_holes: bool = False,
) -> Mask:
    """Extracts a binary mask from an image.

    Args:
        wsi: The WSI object.
        mask_level_idx: The level index of the WSI at which we want to extract the mask.
        kernel_size: The size of the kernel for morphological operations.
        gray_threshold: The threshold for the gray scale image.
        fill_holes: Whether to fill holes in the mask.
    """
    image = wsi.read_region((0, 0), mask_level_idx, wsi.level_dimensions[mask_level_idx])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    gray = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    mask_array = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)

    if fill_holes:
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        contour, _ = cv2.findContours(mask_array, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask_array, [cnt], 0, (1,), -1)

    scale_factors = (
        wsi.level_dimensions[0][0] / wsi.level_dimensions[mask_level_idx][0],
        wsi.level_dimensions[0][1] / wsi.level_dimensions[mask_level_idx][1],
    )

    return Mask(mask_array=mask_array, mask_level_idx=mask_level_idx, scale_factors=scale_factors)


def get_mask_level(
    wsi: Wsi,
    width: int,
    height: int,
    target_mpp: float,
    min_mask_patch_pixels: int = 3 * 3,
) -> int:
    """For performance reasons, we generate the mask at the lowest resolution level possible.

    However, if minimum resolution level has too few pixels, the patches scaled to that level will
    be too small or even collapse to a single pixel. This function allows to find the lowest
    resolution level that yields mask patches with at least `min_mask_patch_pixels` pixels.

    Args:
        wsi: The WSI object.
        width: The width of the patches to be extracted, in pixels (at target_mpp).
        height: The height of the patches to be extracted, in pixels.
        target_mpp: The target microns per pixel (mpp) for the patches.
        min_mask_patch_pixels: The minimum number of pixels required for the mask patches.
            Mask patch refers to width / height at target_mpp scaled down to the WSI level
            at which the mask is generated.
    """
    level_mpps = wsi.mpp * np.array(wsi.level_downsamples)
    mask_level_idx = None

    for level_idx, level_mpp in reversed(list(enumerate(level_mpps))):
        mpp_ratio = target_mpp / level_mpp
        scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)

        if scaled_width * scaled_height >= min_mask_patch_pixels:
            mask_level_idx = level_idx
            break

    if mask_level_idx is None:
        raise ValueError("No level with the specified minimum number of patch pixels available.")

    return mask_level_idx
