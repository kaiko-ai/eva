"""Image I/O related functions."""

import os

import cv2
import numpy as np
import numpy.typing as npt


def read_image(path: str) -> npt.NDArray[np.uint8]:
    """Reads the image from a file path as a RGB.

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
    """Loads an image file as a numpy array.

    Args:
        path: The path to the image file.
        flags: Specifies the way in which the image should be read.

    Returns:
        The image as a numpy array.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        IOError: If the image could not be loaded.
    """
    if not _is_file(path):
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

    if image.ndim == 2 and flags == cv2.IMREAD_COLOR:
        image = image[:, :, np.newaxis]

    return np.asarray(image).astype(np.uint8)


def _is_file(path: str) -> bool:
    """Checks if the input path is a valid file.

    Args:
        path: The file path to be checked.

    Returns:
        A boolean value whether the file exists.
    """
    return os.path.exists(path) and os.stat(path).st_size != 0 and os.path.isfile(path)
