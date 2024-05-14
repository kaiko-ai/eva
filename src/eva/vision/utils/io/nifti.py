"""NIfTI I/O related functions."""

from typing import Any, Tuple

import nibabel as nib
import numpy.typing as npt

from eva.vision.utils.io import _utils


def read_nifti(
    path: str, slice_index: int | None = None, *, use_storage_dtype: bool = True
) -> npt.NDArray[Any]:
    """Reads and loads a NIfTI image from a file path.

    Args:
        path: The path to the NIfTI file.
        slice_index: Whether to read only a slice from the file.
        use_storage_dtype: Whether to cast the raw image
            array to the inferred type.

    Returns:
        The image as a numpy array (height, width, channels).

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image_data = nib.load(path)  # type: ignore
    if slice_index is not None:
        image_data = image_data.slicer[:, :, slice_index : slice_index + 1]  # type: ignore

    image_array = image_data.get_fdata()  # type: ignore
    if use_storage_dtype:
        image_array = image_array.astype(image_data.get_data_dtype())  # type: ignore
    return image_array


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
    image_shape = fetch_nifti_shape(path)
    return image_shape[-1]


def fetch_nifti_shape(path: str) -> Tuple[int]:
    """Fetches the NIfTI image shape from a file.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The image shape.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image = nib.load(path)  # type: ignore
    return image.header.get_data_shape()  # type: ignore
