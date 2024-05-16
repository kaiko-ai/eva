"""Image conversion related functionalities."""

from typing import Any

import numpy as np
import numpy.typing as npt


def to_8bit(image: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Casts an image array of higher bit image (i.e. 16bit) to 8bit.

    Args:
        image: The image array to convert.

    Returns:
        The image as uint8 (0, 255) array.
    """
    image_scaled = image - image.min()
    image_scaled /= image_scaled.max()
    image_scaled *= 255
    return image_scaled.astype(np.uint8)
