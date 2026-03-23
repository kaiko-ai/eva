"""mat I/O related functions."""

import os
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


def save_mat(path: str, data: Dict[str, npt.NDArray[Any]]) -> None:
    """Saves a mat file.

    Args:
        path: The path to save the mat file.
        data: The dictionary containing the data to save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scipy.io.savemat(path, data)
