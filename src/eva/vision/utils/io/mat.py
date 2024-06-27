"""mat I/O related functions."""

from typing import Any, Dict

import numpy.typing as npt
import scipy.io

from eva.vision.utils.io import _utils


def read_mat(path: str) -> Dict[str, npt.NDArray[Any]]:
    """Reads and loads a mat file.

    Args:
        path: The path to the mat file.

    Returns:
        mat file as dictionary.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
    """
    _utils.check_file(path)
    return scipy.io.loadmat(path)
