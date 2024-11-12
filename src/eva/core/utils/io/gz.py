"""Utils for .gz files."""

import gzip
import os


def gunzip_file(path: str, unpack_dir: str | None = None, keep: bool = True) -> str:
    """Unpacks a .gz file to the provided directory.

    Args:
        path: Path to the .gz file to extract.
        unpack_dir: Directory to extract the file to. If `None`, it will use the
            same directory as the compressed file.
        keep: Whether to keep the compressed .gz file.

    Returns:
        The path to the extracted file.
    """
    unpack_dir = unpack_dir or os.path.dirname(path)
    os.makedirs(unpack_dir, exist_ok=True)
    save_path = os.path.join(unpack_dir, os.path.basename(path).replace(".gz", ""))
    if not os.path.isfile(save_path):
        with gzip.open(path, "rb") as f_in:
            with open(save_path, "wb") as f_out:
                f_out.write(f_in.read())
        if not keep:
            os.remove(path)
    return save_path
