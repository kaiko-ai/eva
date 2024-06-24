"""Dataset validation related functions."""

import os

from typing_extensions import List, Tuple

from eva.vision.data.datasets import vision

_SUFFIX_ERROR_MESSAGE = "Please verify that the data are properly downloaded and stored."
"""Common suffix dataset verification error message."""


def check_dataset_integrity(
    dataset: vision.VisionDataset,
    *,
    length: int | None,
    n_classes: int,
    first_and_last_labels: Tuple[str, str],
) -> None:
    """Verifies the datasets integrity.

    Raise:
        ValueError: If the input dataset's values do not
            match the expected ones.
    """
    if length and len(dataset) != length:
        raise ValueError(
            f"Dataset's '{dataset.__class__.__qualname__}' length "
            f"({len(dataset)}) does not match the expected one ({length}). "
            f"{_SUFFIX_ERROR_MESSAGE}"
        )

    dataset_classes: List[str] = getattr(dataset, "classes", [])
    if dataset_classes and len(dataset_classes) != n_classes:
        raise ValueError(
            f"Dataset's '{dataset.__class__.__qualname__}' number of classes "
            f"({len(dataset_classes)}) does not match the expected one ({n_classes})."
            f"{_SUFFIX_ERROR_MESSAGE}"
        )

    if dataset_classes and (dataset_classes[0], dataset_classes[-1]) != first_and_last_labels:
        raise ValueError(
            f"Dataset's '{dataset.__class__.__qualname__}' first and last labels "
            f"({(dataset_classes[0], dataset_classes[-1])}) does not match the expected "
            f"ones ({first_and_last_labels}). {_SUFFIX_ERROR_MESSAGE}"
        )


def check_dataset_exists(dataset_dir: str, download_available: bool) -> None:
    """Verifies that the dataset folder exists.

    Raise:
        FileNotFoundError: If the dataset folder does not exist.
    """
    if not os.path.isdir(dataset_dir):
        error_message = f"Dataset not found at '{dataset_dir}'."
        if download_available:
            error_message += " You can set `download=True` to download the dataset automatically."
        raise FileNotFoundError(error_message)


def check_number_of_files(file_paths: List[str], expected_length: int, split: str | None) -> None:
    """Verifies the number of files in the dataset.

    Raise:
        ValueError: If the number of files in the dataset does not match the expected one.
    """
    if len(file_paths) != expected_length:
        raise ValueError(
            f"Expected {expected_length} files, for split '{split}' found {len(file_paths)}. "
            f"{_SUFFIX_ERROR_MESSAGE}"
        )
