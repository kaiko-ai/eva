"""NIfTI I/O related functions."""

import nibabel as nib
import numpy as np
import numpy.typing as npt

from eva.vision.utils.io import _utils


def read_nifti(path: str, ct_slice: int | None = None) -> npt.NDArray[np.uint8]:
    """Reads a NIfTI image from a file path.

    Args:
        path: The path to the NIfTI file.
        ct_slice: Whether to return a specific CT slice.

    Returns:
        The image as a numpy array.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image = nib.load(path).get_fdata()  # type: ignore
    image_array = np.asarray(image).astype(np.uint8)
    if ct_slice is not None:
        if 0 < ct_slice or ct_slice > image_array.shape[2]:
            raise ValueError("Invalid CT slice index.")
        image_array = image_array[:, :, ct_slice]
    return image_array
