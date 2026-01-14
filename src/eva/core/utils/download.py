"""Download utilities for datasets and other resources.

This module contains utilities for downloading datasets from HuggingFace Hub
and unpacking archive files. The code is adapted from:
kaiko-eng/libs/dagster/kaiko/dagster/huggingface/hf_hub_download.py
kaiko-eng/libs/dagster/kaiko/dagster/file_utils/unpack_archive.py
kaiko-eng/libs/dagster/kaiko/dagster/file_utils/utils.py
"""

import gc
import os
import shutil
from pathlib import Path

import huggingface_hub
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError
from loguru import logger
from requests.exceptions import ChunkedEncodingError, ReadTimeout, Timeout

MAX_TRIES = 10

# Supported archive extensions (native shutil support)
_SUPPORTED_EXTENSIONS = [".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"]

# Unsupported extensions that require additional implementation
_UNSUPPORTED_EXTENSIONS = [".7z", ".gz"]


class HuggingFaceDownloadError(Exception):
    """Raised when an error occurs during Hugging Face repository download."""

    def __init__(self, repo_name: str, message: str):
        """Initialize HuggingFaceDownloadError.

        Args:
            repo_name: Name of the repository that failed to download.
            message: Error message describing what went wrong.
        """
        self.repo_name = repo_name
        super().__init__(message)


def _verify_downloaded_files(
    download_path: Path,
    repo_name: str,
    repo_type: str,
    hf_token: str | None,
    files: list[str] | None,
) -> None:
    """Verify that the requested files were successfully downloaded.

    Uses either the specified files or all files in the repository
    and verifies that all the expected files are present.

    Args:
        download_path: Path to the downloaded repository
        repo_name: Name of the repository to download data from
        repo_type: Type of the repository (dataset/model)
        hf_token: Hugging Face token for API access
        files: List of specific files requested, or None for all files

    Raises:
        HuggingFaceDownloadError: If requested files are missing
    """
    all_asset_files = _filter_relevant_files(list(download_path.rglob("*")))
    actual_files = [f for f in all_asset_files if f.is_file()]

    if not actual_files:
        raise HuggingFaceDownloadError(
            repo_name=repo_name,
            message=f"No files were downloaded from Hugging Face repository '{repo_name}'",
        )

    if files is None:
        try:
            expected_repo_files = huggingface_hub.list_repo_files(
                repo_id=repo_name, repo_type=repo_type, token=hf_token
            )
            expected_files = [
                f for f in expected_repo_files if not (f.startswith((".git", ".cache")))
            ]
        except Exception as e:
            raise HuggingFaceDownloadError(
                repo_name=repo_name,
                message=f"Access denied to Hugging Face repository '{repo_name}', "
                "could not list repo files.",
            ) from e
    else:
        expected_files = files

    actual_file_names = {f.name for f in actual_files}
    actual_file_paths = {str(f.relative_to(download_path)) for f in actual_files}

    missing_files = []
    for expected_file in expected_files:
        if expected_file not in actual_file_names and expected_file not in actual_file_paths:
            missing_files.append(expected_file)

    if missing_files:
        raise HuggingFaceDownloadError(
            repo_name=repo_name,
            message=f"Expected files not found in Hugging Face repository '{repo_name}': "
            f"Expected files not found: {missing_files}",
        )

    logger.info(f"Successfully verified {len(actual_files)} files for repo '{repo_name}'")


def _filter_relevant_files(file_paths: list[Path]) -> list[Path]:
    """Filter out hidden files and cache directories from a list of file paths.

    Args:
        file_paths: List of Path objects to filter

    Returns:
        Filtered list excluding hidden files and cache directories
    """
    return [
        f
        for f in file_paths
        if not any(part.startswith(".") for part in f.parts)
        and not any(part in [".cache", "__pycache__", "node_modules"] for part in f.parts)
    ]


def repo_download(  # noqa: C901
    repo_name: str,
    repo_type: str,
    local_dir: str | Path,
    hf_token: str | None = None,
    files: list[str] | None = None,
    retries: int = MAX_TRIES,
    unpack: bool = True,
) -> Path:
    """Download a repo from Huggingface Hub.

    Args:
        repo_name: The name of the repo
        repo_type: The type of the repo (e.g. "dataset" or "model")
        local_dir: The local directory to download to
        hf_token: The Huggingface Hub token
        files: The files to download. Defaults to None.
        retries: The amount of times download is tried
        unpack: Boolean whether to unpack archives or not

    Returns:
        The path to the downloaded repo.
    """
    local_dir = Path(local_dir)

    # Fail if directory exists and is not empty to avoid unpredictable behavior
    if local_dir.exists() and any(local_dir.iterdir()):
        raise HuggingFaceDownloadError(
            repo_name=repo_name,
            message=f"Download directory '{local_dir}' already exists and is not empty. "
            "Please remove or empty the directory or specify a different location.",
        )

    local_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = local_dir / ".cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    path = None
    tries = 0
    while tries < retries:
        try:
            tries += 1

            path = huggingface_hub.snapshot_download(
                repo_name,
                repo_type=repo_type,
                token=hf_token,
                cache_dir=hf_cache_dir,
                local_dir=local_dir,
                ignore_patterns=[".git*"],
                allow_patterns=files,
                etag_timeout=30.0,
                max_workers=4,
            )

            # Verify the download was successful
            if path and Path(path).exists():
                _verify_downloaded_files(Path(path), repo_name, repo_type, hf_token, files)
            else:
                raise HuggingFaceDownloadError(
                    repo_name=repo_name,
                    message=f"Download failed for repository '{repo_name}': No valid path returned",
                )
            break
        except (
            GatedRepoError,
            RepositoryNotFoundError,
        ) as e:
            raise HuggingFaceDownloadError(
                repo_name=repo_name,
                message=f"Access denied to Hugging Face repository '{repo_name}': {str(e)}",
            ) from e
        except (
            ChunkedEncodingError,
            ReadTimeout,
            Timeout,
            LocalEntryNotFoundError,
        ) as e:
            logger.warning(
                f"Download error ({type(e).__name__}): Retrying... {tries} / {MAX_TRIES}"
            )
            if tries >= retries:
                raise Exception(f"Failed to download after {retries} retries: {str(e)}") from e
        except Exception as e:
            raise e

    if path is None:
        raise Exception("Failed to download the repo")

    # Remove cache dir before upload:
    if os.path.exists(hf_cache_dir):
        shutil.rmtree(hf_cache_dir)

    # Unpacking compressed files
    if unpack:
        download_path = Path(path)
        all_downloads_paths = _filter_relevant_files(list(download_path.rglob("*")))
        for archive_path in all_downloads_paths:
            if _should_unpack(archive_path):
                logger.info(f"Unpacking archive: {archive_path}")
                unpack_archive(archive_path)

    return Path(path)


# --- Unpack archive utilities ---


def _get_archive_suffix(file_path: Path) -> str:
    """Returns the archive suffix for a file path, handling compound extensions like .tar.gz."""
    suffixes = file_path.suffixes
    if len(suffixes) >= 2 and suffixes[-2:] == [".tar", ".gz"]:
        return ".tar.gz"
    if len(suffixes) >= 2 and suffixes[-2:] == [".tar", ".bz2"]:
        return ".tar.bz2"
    if len(suffixes) >= 2 and suffixes[-2:] == [".tar", ".xz"]:
        return ".tar.xz"
    return file_path.suffix


def _should_unpack(file_path: Path | str) -> bool:
    """Returns True if the file should be unpacked, False otherwise."""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.is_file():
        return False

    archive_suffix = _get_archive_suffix(file_path)

    # Check for unsupported formats and raise an error
    if archive_suffix in _UNSUPPORTED_EXTENSIONS:
        # Special case: .nii.gz files are not archives, they're compressed NIfTI images
        if str(file_path).endswith(".nii.gz"):
            return False
        raise NotImplementedError(
            f"Unpacking '{archive_suffix}' files is not yet implemented. "
            f"File: {file_path}. "
            "Please implement support for this format in eva/core/utils/download.py"
        )

    return archive_suffix in _SUPPORTED_EXTENSIONS


def unpack_archive(  # noqa: C901
    archive_path: Path,
    unpack_dir: Path | None = None,
    ignore_macos: bool = True,
    ignore_ds_store: bool = True,
    remove_archive: bool = True,
    max_depth: int = 5,
    _current_depth: int = 0,
) -> Path:
    """Recursively unpacks archive files (zip, tar, tar.gz, etc.) to a specified directory.

    This function handles various archive formats including .zip, .tar, .tar.gz, .tar.bz2,
    and .tar.xz files. It can recursively unpack nested archives up to a specified depth
    and automatically cleans up system files and redundant subdirectories.

    Notes:
        - Automatically removes redundant subdirectories (e.g., /path/myarchive/myarchive/).
        - Memory management is handled through garbage collection after unpacking.
        - .7z and standalone .gz files are NOT supported and will raise NotImplementedError.

    Args:
        archive_path: Path to the archive file to unpack.
        unpack_dir: Directory where the archive will be unpacked. If None, creates
            directory with same name as archive in same location.
        ignore_macos: If True, removes macOS-specific files (e.g., __MACOSX directories)
            during unpacking.
        ignore_ds_store: If True, removes .DS_Store files during unpacking.
        remove_archive: If True, deletes the original archive file after successful unpacking.
        max_depth: Maximum depth for recursive unpacking of nested archives. Prevents
            infinite recursion in case of circular references.
        _current_depth: Internal parameter tracking current recursion depth.

    Returns:
        Path: Path to the directory containing the unpacked contents.

    Raises:
        ValueError: If the recursive unpacking exceeds the maximum depth.
        NotImplementedError: If the archive format is not supported (.7z, .gz).
    """
    if _current_depth > max_depth:
        raise ValueError(f"Maximum unpacking depth of {max_depth} exceeded")

    unpacked_path = unpack_dir or archive_path.with_suffix("")
    # Handle compound extensions like .tar.gz
    if unpacked_path.suffix in [".tar"]:
        unpacked_path = unpacked_path.with_suffix("")
    unpacked_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Unpacking {archive_path} to {unpacked_path}")
    shutil.unpack_archive(archive_path, unpacked_path)

    gc.collect()
    _cleanup_system_files(unpacked_path, ignore_macos, ignore_ds_store)
    _remove_redundant_subdir(unpacked_path)

    if remove_archive:
        archive_path.unlink()

    # Recursively check for further files to unpack
    for unpacked_file in unpacked_path.rglob("*"):
        if _should_unpack(unpacked_file):
            unpack_archive(
                unpacked_file,
                _current_depth=_current_depth + 1,
            )

    return Path(unpacked_path)


# --- File cleanup utilities ---


def _safe_remove(path: Path):
    try:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            logger.info(f"Tried to remove {path}, but does not exist.")
    except Exception as e:
        logger.error(f"Error while deleting {path}: {e}")


def _cleanup_system_files(
    root: Path, remove_macos: bool = True, remove_ds_store: bool = True
) -> None:
    """Removes MacOS and .DS_Store files from the provided root directory."""
    for path in root.rglob("*"):
        if remove_macos and path.name.startswith("__MACOSX"):
            _safe_remove(path)
        elif remove_ds_store and path.name.startswith(".DS_Store"):
            _safe_remove(path)


def _remove_redundant_subdir(path: Path) -> None:
    """Remove redundant subdirectory created during archive unpacking.

    For archives tmp/myarchive.zip where there is a "myarchive" folder within the archive,
    unpacking can result in /tmp/myarchive/myarchive/. This function removes the
    redundant subfolder.
    """
    if path.is_dir():
        subdirs = list(path.iterdir())
        if len(subdirs) == 1 and subdirs[0].is_dir() and subdirs[0].name == path.name:
            for item in subdirs[0].iterdir():
                dest = path / item.name
                if dest.exists():
                    logger.warning(f"Skipping move of {item} because {dest} already exists.")
                    continue
                shutil.move(str(item), str(path))
            if not any(subdirs[0].iterdir()):
                try:
                    subdirs[0].rmdir()
                except Exception as e:
                    logger.warning(f"Failed to remove {subdirs[0]}: {e}")
            else:
                logger.warning(f"Did not remove {subdirs[0]} because it is not empty.")


# --- Convenience wrapper ---


def download_from_huggingface(
    repo_id: str,
    local_dir: str,
    repo_type: str = "dataset",
    files: list[str] | None = None,
    unpack: bool = True,
) -> str:
    """Downloads a HuggingFace repository to a local directory.

    This is a convenience wrapper around repo_download. Authentication is handled
    automatically via the HF_TOKEN environment variable or `huggingface-cli login`.

    Args:
        repo_id: The repository ID on HuggingFace (e.g., "jamessyx/PathMMU").
        local_dir: The local directory to download the repository to.
        repo_type: The type of repository ("dataset" or "model").
        files: List of specific files to download. If None, downloads all files.
        unpack: Whether to extract archive files after download.

    Returns:
        The path to the downloaded repository.
    """
    path = repo_download(
        repo_name=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        files=files,
        unpack=unpack,
    )
    return str(path)
