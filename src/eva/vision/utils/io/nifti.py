"""NIfTI I/O related functions."""

import nibabel as nib
import numpy as np
import numpy.typing as npt

from eva.vision.utils.io import _utils


def read_nifti(path: str, slice_index: int | None = None) -> npt.NDArray[np.uint8]:
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
