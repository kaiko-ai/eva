"""NIfTI I/O related functions."""

from typing import Any, Tuple

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel import orientations

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
    image_data: nib.Nifti1Image = nib.load(path)  # type: ignore
    if slice_index is not None:
        image_data = image_data.slicer[:, :, slice_index : slice_index + 1]

    image_array = image_data.get_fdata()
    if use_storage_dtype:
        image_array = image_array.astype(image_data.get_data_dtype())

    return image_array


def save_array_as_nifti(
    array: npt.ArrayLike,
    filename: str,
    *,
    dtype: npt.DTypeLike | None = np.int64,
) -> None:
    """Saved a numpy array as a NIfTI image file.

    Args:
        array: The image array to save.
        filename: The name to save the image like.
        dtype: The data type to save the image.
    """
    nifti_image = nib.Nifti1Image(array, affine=np.eye(4), dtype=dtype)  # type: ignore
    nifti_image.to_filename(filename)


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


def fetch_nifti_axis_direction_code(path: str) -> str:
    """Fetches the NIfTI axis direction code from a file.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The axis direction codes as string (e.g. "LAS").
    """
    _utils.check_file(path)
    image_data: nib.Nifti1Image = nib.load(path)  # type: ignore
    return "".join(orientations.aff2axcodes(image_data.affine))
