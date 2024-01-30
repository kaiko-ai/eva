"""File IO utilities and helper functions."""

import gzip
import os
import shutil
from urllib import request
from urllib.parse import urlparse


def make_folder_with_permissions(path: str, mode: int = 0o755):
    """Create a folder along with its parent directories and set permissions.

    Parameters:
        path: The path to the folder to be created.
        mode: The permission mode to set for the folder.
    """
    os.makedirs(path, exist_ok=True)
    os.chmod(path, mode)


def download_data(url: str, output_dir: str = "") -> str:
    """Downloads and saves a file from the specified URL.

    Args:
        url: The URL of the file to be downloaded.
        output_dir: The directory where the downloaded file will be saved.
            If not provided, the file will be saved in the current working directory.

    Returns:
        The relative path to the downloaded file.
    """
    if not url.startswith("http"):
        raise ValueError("Please provide only 'http' addresses.")

    filename = os.path.basename(urlparse(url).path)
    output_file = os.path.join(output_dir, filename)
    request.urlretrieve(url, output_file)  # nosec
    return output_file


def extract_and_replace(filename: str) -> None:
    """Extracts content from a gzip-compressed file and replaces the original file.

    Args:
        filename: The name of the gzip-compressed file.
    """
    with gzip.open(filename, "rb") as gz_file:
        with open(filename.removesuffix(".gz"), "wb") as file:
            shutil.copyfileobj(gz_file, file)

    os.remove(filename)


def download_and_extract_archive(url: str, output_dir: str = "") -> None:
    """Downloads and decompresses a file from the specified URL.

    Args:
        url: The URL of the gzip-compressed file to be downloaded.
        output_dir: The directory where the downloaded file will be saved.
            If not provided, the file will be saved in the current working directory.

    Returns:
        The relative path to the downloaded file.
    """
    make_folder_with_permissions(output_dir)
    output_file = download_data(url, output_dir)
    extract_and_replace(output_file)


def verify_file(filename: str, expected_bytes: int) -> bool:
    """Verifies if a file exists and has the expected size.

    Args:
        filename: The path to the file.
        expected_bytes: The expected size of the file in bytes.

    Returns:
        A boolean indicating Whether the file exists and is valid.
    """
    return os.path.isfile(filename) and os.path.getsize(filename) == expected_bytes
