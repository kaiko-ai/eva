"""mat I/O related functions."""

import scipy.io
from typing import Any, Dict

from eva.vision.utils.io import _utils


def read_mat(path: str) -> Dict[Any, Any]:
    """Reads and loads a NIfTI image from a file path.

    Args:
        path: The path to the mat file.

    Returns:
        mat file as dictionary.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
    """
    _utils.check_file(path)
    return scipy.io.loadmat(path)
