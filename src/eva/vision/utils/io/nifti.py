"""NIfTI I/O related functions."""

import nibabel as nib
import numpy as np
import numpy.typing as npt

from eva.vision.utils.io import _utils


def read_nifti(path: str, channel: int | None = None) -> npt.NDArray[np.uint8]:
    """Reads a NIfTI image from a file path.

    Args:
        path: The path to the NIfTI file.
        channel: Whether to return a specific image channel (slice).

    Returns:
        The image as a numpy array.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image = nib.load(path).get_fdata()  # type: ignore
    image_array = np.asarray(image).astype(np.uint8)
    if channel is not None:
        if 0 < channel or channel > image_array.shape[2]:
            raise ValueError("Invalid channel index.")
        image_array = image_array[:, :, channel]
    return image_array
