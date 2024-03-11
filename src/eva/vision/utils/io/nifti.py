"""NIfTI I/O related functions."""
from typing import Tuple
import nibabel as nib
import numpy as np
import numpy.typing as npt

from eva.vision.utils.io import _utils


def read_nifti_slice(path: str, slice_index: int, *, dtype: type = np.uint8) -> npt.NDArray[np.uint8]:
    """Reads a NIfTI image from a file path.

    Args:
        path: The path to the NIfTI file.
        slice_index: The image slice index to return. If `None`, it will
            return the full 3D image.
        dtype: The image byte type to cast the image array.

    Returns:
        The image as a numpy array.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image_data = nib.load(path)
    image_slice = image_data.slicer[:, :, slice_index:slice_index + 1]
    raw_image_array = image_slice.get_fdata()
    image_array = raw_image_array.astype(dtype)
    return np.squeeze(image_array)


def fetch_total_nifti_slices(path: str) -> int:
    """Fetches the total slides of a NIfTI image file.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The number of the total available slides.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image = nib.load(path)
    image_shape = image.header.get_data_shape()
    return image_shape[-1]
