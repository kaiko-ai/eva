"""NIfTI I/O related functions."""

from typing import Any

import nibabel as nib
import numpy.typing as npt

from eva.vision.utils.io import _utils


def read_nifti_slice(path: str, slice_index: int) -> npt.NDArray[Any]:
    """Reads and loads a NIfTI image from a file path as `uint8`.

    Args:
        path: The path to the NIfTI file.
        slice_index: The image slice index to return. If `None`, it will
            return the full 3D image.

    Returns:
        The image as a numpy array.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image_data = nib.load(path)  # type: ignore
    dtype = image_data.get_data_dtype()  # type: ignore
    image_slice = image_data.slicer[:, :, slice_index : slice_index + 1]  # type: ignore
    image_array = image_slice.get_fdata()
    return image_array.astype(dtype)


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
    image = nib.load(path)  # type: ignore
    image_shape = image.header.get_data_shape()  # type: ignore
    return image_shape[-1]
