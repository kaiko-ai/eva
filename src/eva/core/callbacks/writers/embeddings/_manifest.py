"""Manifest file manager."""

import csv
import io
import os
from typing import Any, Dict, List

import _csv
import torch


class ManifestManager:
    """Class for writing the embedding manifest files."""

    def __init__(
        self,
        output_dir: str,
        metadata_keys: List[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Initializes the writing manager.

        Args:
            output_dir: The directory where the embeddings will be saved.
            metadata_keys: An optional list of keys to extract from the batch
                metadata and store as additional columns in the manifest file.
            overwrite: Whether to overwrite the output directory.
        """
        self._output_dir = output_dir
        self._metadata_keys = metadata_keys or []
        self._overwrite = overwrite

        self._manifest_file: io.TextIOWrapper
        self._manifest_writer: _csv.Writer  # type: ignore

        self._setup()

    def _setup(self) -> None:
        """Initializes the manifest file and sets the file object and writer."""
        manifest_path = os.path.join(self._output_dir, "manifest.csv")
        if os.path.exists(manifest_path) and not self._overwrite:
            raise FileExistsError(
                f"A manifest file already exists at {manifest_path}, which indicates that the "
                "chosen output directory has been previously used for writing embeddings."
            )
        self._manifest_file = open(manifest_path, "w", newline="")
        self._manifest_writer = csv.writer(self._manifest_file)
        self._manifest_writer.writerow(
            ["origin", "embeddings", "target", "split"] + self._metadata_keys
        )

    def update(
        self,
        input_name: str,
        save_name: str,
        target: str,
        split: str | None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Adds a new entry to the manifest file."""
        metadata_entries = _to_dict_values(metadata or {})
        self._manifest_writer.writerow([input_name, save_name, target, split] + metadata_entries)

    def close(self) -> None:
        """Closes the manifest file."""
        if self._manifest_file:
            self._manifest_file.close()


def _to_dict_values(data: Dict[str, Any]) -> List[Any]:
    return [value.item() if isinstance(value, torch.Tensor) else value for value in data.values()]
