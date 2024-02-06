"""Image IO utils."""

import os

import cv2
import nibabel as nib
import numpy as np
import numpy.typing as npt


def _load_image_file(path: str, flags: int = cv2.IMREAD_COLOR) -> npt.NDArray[np.uint8]:
    """Loads an image file as a numpy array.

    Args:
        path: The path to the image file.
        flags: Specifies the way in which the image should be read.

    Returns:
        The image as a numpy.ndarray.
    """
    if not os.path.isfile(path):
        raise FileExistsError(f"File '{path}' does not exist.")

    image = cv2.imread(path, flags=flags)
    if image is None:
        raise IOError(f"'{path}' could not be loaded. Please verify that it's a valid image file.")

    return np.asarray(image).astype(np.uint8)


def load_image(path: str, as_rgb: bool = True) -> npt.NDArray[np.uint8]:
    """Reads an image from a file path as a RGB.

    Args:
        path: The path to the image file.
        as_rgb: If True, the image is converted to RGB.

    Returns:
        The image as a RGB numpy.ndarray.
    """
    image = _load_image_file(path, flags=cv2.IMREAD_COLOR)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if as_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.asarray(image).astype(np.uint8)


def load_nifti_image(path: str) -> npt.NDArray[np.uint8]:
    """Reads a NIfTI image from a file path.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The image as a numpy.ndarray.
    """
    image = nib.load(path).get_fdata()
    return np.asarray(image).astype(np.uint8)


def load_nifti_image_slice(path: str, slice_: int) -> npt.NDArray[np.uint8]:
    """Reads a NIfTI image slice from a file path.

    Args:
        path: The path to the NIfTI file.
        slice_: The slice to extract from the image.

    Returns:
        The image as a numpy.ndarray.
    """
    image = load_nifti_image(path)

    if slice_ < 0 or slice_ >= image.shape[-1]:
        raise ValueError(f"Invalid slice index. The image has {image.shape[2]} slices.")

    return image[:, :, slice_]
