"""NIfTI I/O related functions."""

import nibabel as nib
import numpy as np
import numpy.typing as npt

from eva.vision.utils.io import _utils


def read_nifti_slice_from_image(path: str, slice_index: int | None = None) -> npt.NDArray[np.uint8]:
    """Reads a NIfTI image from a file path.

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
    image = nib.load(path).get_fdata()  # type: ignore
    image_array = np.asarray(image).astype(np.uint8)
    if slice_index is not None:
        image_array = image_array[:, :, slice_index]
    return image_array


def read_nifti_slice_from_image_with_unknown_dimension(
    path: str, slice_number: int, max_slice_number: int
) -> npt.NDArray[np.uint8]:
    """Reads one NIfTI image from a file path based on a slice number.

    Since the total number of slices in the 3D image is unknown, before
    reading the image, the function calculates the slide index based on
    the provided slice number and the maximum number of slices to consider.

    Args:
        path: The path to the NIfTI file.
        slice_number: The slice number to return.
        max_slice_number: The maximum number of slices to consider.

    Returns:
        The image as a numpy array.
    """
    _utils.check_file(path)
    image = nib.load(path).get_fdata()  # type: ignore
    image_array = np.asarray(image).astype(np.uint8)
    n_slices = image_array.shape[-1]

    if n_slices <= max_slice_number:
        raise Warning(
            f"The number of slices in the image is less than the maximum number of "
            f"slices to consider: {n_slices} <= {max_slice_number}"
        )

    first_slice_index = int(0.5 * n_slices / max_slice_number)

    slice_index = first_slice_index + (n_slices - 1) * slice_number / max_slice_number
    slice_index = min(int(slice_index), n_slices - 1)

    return image_array[:, :, slice_index]
