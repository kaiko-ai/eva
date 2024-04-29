"""Image conversion related functionalities."""

from typing import Any

import numpy as np
import numpy.typing as npt


def to_8bit(image_array: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Casts an image of higher bit image (i.e. 16bit) to 8bit.

    Args:
        image_array: The image array to convert.

    Returns:
        The image as normalized as a 8-bit format.
    """
    if np.issubdtype(image_array.dtype, np.integer):
        image_array = image_array.astype(np.float64)

    image_scaled_array = image_array - image_array.min()
    image_scaled_array /= image_scaled_array.max()
    image_scaled_array *= 255
    return image_scaled_array.astype(np.uint8)
