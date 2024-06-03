"""Image I/O related functions."""

import cv2
import numpy as np
import numpy.typing as npt
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional

from eva.vision.utils.io import _utils


def read_image(path: str) -> npt.NDArray[np.uint8]:
    """Reads and loads the image from a file path as a RGB.

    Args:
        path: The path of the image file.

    Returns:
        The RGB image as a numpy array (HxWxC).

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        IOError: If the image could not be loaded.
    """
    return read_image_as_array(path, cv2.IMREAD_COLOR)


def read_image_as_tensor(path: str) -> tv_tensors.Image:
    """Reads and loads the image from a file path as a RGB torch tensor.

    Args:
        path: The path of the image file.

    Returns:
        The RGB image as a torch tensor (CxHxW).

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        IOError: If the image could not be loaded.
    """
    image_array = read_image(path)
    return functional.to_image(image_array)


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

    return np.asarray(image, dtype=np.uint8)
