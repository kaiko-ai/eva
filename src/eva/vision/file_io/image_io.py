"""Image IO utils."""
import os

import cv2
import numpy as np
import numpy.typing as npt


def load_image_file_as_array(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> npt.NDArray[np.uint8]:
    """Loads an image file as a RGB or grayscale numpy array.

    Args:
        path: The path to the image file.
        flags: Specifies the way in which the image should be read.
            The default is cv2.IMREAD_UNCHANGED.

    Returns:
        The image as a numpy.ndarray.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        IOError: If the image could not be loaded.
    """
    if not os.path.isfile(path):
        raise FileExistsError(
            f"Input '{path if isinstance(path, str) else type(path)}' "
            "could not be recognized as a valid file. Please verify "
            "that the file exists and is reachable."
        )

    image = cv2.imread(path, flags=flags)
    if image is None:
        raise IOError(
            f"Input '{path}' could not be loaded. "
            "Please verify that the path is a valid image file."
        )

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    return np.asarray(image).astype(np.uint8)
