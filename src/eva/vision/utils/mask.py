import cv2
import numpy as np

from eva.vision.data.wsi.backends.base import Wsi


def get_mask(
    wsi: Wsi,
    level_idx: int,
    kernel_size: tuple[int, int] = (7, 7),
    gray_threshold: int = 220,
    fill_holes: bool = False,
) -> tuple[np.ndarray, float]:
    """Extracts a binary mask from an image.

    Args:
        wsi: The WSI object.
        level_idx: The level index to extract the mask from.
        kernel_size: The size of the kernel for morphological operations.
        gray_threshold: The threshold for the gray scale image.
        fill_holes: Whether to fill holes in the mask.
    """
    image = np.array(
        wsi.read_region([0, 0], len(wsi.level_dimensions) - 1, wsi.level_dimensions[-1])
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
