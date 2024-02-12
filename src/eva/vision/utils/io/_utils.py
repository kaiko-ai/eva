"""File IO utilities."""

import os


def is_file(path: str) -> bool:
    """Checks if the input path is a valid file.

    Args:
        path: The file path to be checked.

    Returns:
        A boolean value whether the file exists.
    """
    return os.path.exists(path) and os.stat(path).st_size != 0 and os.path.isfile(path)


def check_file(path: str) -> None:
    """Checks whether the input path is a valid file and raises and error.

    Args:
        path: The file path to be checked.
    """
    if not is_file(path):
        raise FileExistsError(
            f"Input '{path if isinstance(path, str) else type(path)}' "
            "could not be recognized as a valid file. Please verify "
            "that the file exists and is reachable."
        )
