# type: ignore
"""NIfTI I/O related functions."""

from typing import Any, Tuple

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel import orientations

from eva.core.utils.suppress_logs import SuppressLogs
from eva.vision.utils.io import _utils


def read_nifti(
    path: str,
    slice_index: int | None = None,
    *,
    orientation: str | None = None,
    orientation_reference: str | None = None,
) -> nib.nifti1.Nifti1Image:
    """Reads and loads a NIfTI image from a file path.

    Args:
        path: The path to the NIfTI file.
        slice_index: Whether to read only a slice from the file.
        orientation: The orientation code to reorient the nifti image.
        orientation_reference: Path to a NIfTI file which
            will be used as a reference for the orientation
            transform in case the file missing the pixdim array
            in the NIfTI header.
        use_storage_dtype: Whether to cast the raw image
            array to the inferred type.

    Returns:
        The NIfTI image class instance.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image_data = _load_nifti_silently(path)
    if slice_index is not None:
        image_data = image_data.slicer[:, :, slice_index : slice_index + 1]
    if orientation:
        image_data = _reorient(
            image_data, orientation=orientation, reference_file=orientation_reference
        )

    return image_data


def nifti_to_array(nii: nib.Nifti1Image, use_storage_dtype: bool = True) -> npt.NDArray[Any]:
    """Converts a NIfTI image to a numpy array.

    Args:
        nii: The input NIfTI image.
        use_storage_dtype: Whether to cast the raw image
            array to the inferred type.

    Returns:
        The image as a numpy array (height, width, channels).
    """
    image_array = nii.get_fdata()
    if use_storage_dtype:
        image_array = image_array.astype(nii.get_data_dtype())
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
    nifti_image = nib.Nifti1Image(array, affine=np.eye(4), dtype=dtype)
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
    nii = _load_nifti_silently(path)
    return nii.header.get_data_shape()  # type: ignore


def fetch_nifti_orientation(path: str) -> npt.NDArray[Any]:
    """Fetches the NIfTI image orientation.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The array orientation.
    """
    _utils.check_file(path)
    nii = _load_nifti_silently(path)
    return nib.io_orientation(nii.affine)


def fetch_nifti_axis_direction_code(path: str) -> str:
    """Fetches the NIfTI axis direction code from a file.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The axis direction codes as string (e.g. "LAS").
    """
    _utils.check_file(path)
    image_data: nib.Nifti1Image = nib.load(path)
    return "".join(orientations.aff2axcodes(image_data.affine))


def _load_nifti_silently(path: str) -> nib.Nifti1Image:
    """Reads a NIfTI image in silent mode."""
    with SuppressLogs():
        return nib.load(path)
    raise ValueError(f"Failed to load NIfTI file: {path}")


def _reorient(
    nii: nib.Nifti1Image,
    /,
    orientation: str | tuple[str, str, str] = "RAS",
    reference_file: str | None = None,
) -> nib.Nifti1Image:
    """Reorients a NIfTI image to a specified orientation.

    Args:
        nii: The input NIfTI image.
        orientation: Desired orientation expressed as a
            three-character string (e.g., "RAS") or a tuple
            (e.g., ("R", "A", "S")).
        reference_file: Path to a reference NIfTI file whose
            orientation should be used if the input image lacks
            a valid affine transformation.

    Returns:
        The reoriented NIfTI image.
    """
    affine_matrix, _ = nii.get_qform(coded=True)
    orig_ornt = (
        fetch_nifti_orientation(reference_file)
        if reference_file and affine_matrix is None
        else nib.io_orientation(nii.affine)
    )
    targ_ornt = orientations.axcodes2ornt(orientation)
    transform = orientations.ornt_transform(orig_ornt, targ_ornt)
    return nii.as_reoriented(transform)
