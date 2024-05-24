"""Manifest file manager."""

import csv
import os
from typing import Any, Dict, List


class ManifestManager:
    """Class for writing the embedding manifest files."""

    def __init__(
        self, output_dir: str, metadata_keys: List[str] | None = None, overwrite: bool = False
    ):
        self._output_dir = output_dir
        self._metadata_keys = metadata_keys or []
        self._overwrite = overwrite

        self._init_manifest()

    def _init_manifest(self) -> None:
        """Initializes the manifest file and sets the file object and writer."""
        manifest_path = os.path.join(self._output_dir, "manifest.csv")
        if os.path.exists(manifest_path) and not self._overwrite:
            raise FileExistsError(
                f"Manifest file already exists at {manifest_path}. This likely means that the "
                "embeddings have been computed before. Consider using `eva fit` instead "
                "of `eva predict_fit` or `eva predict`."
            )
        self._manifest_file = open(manifest_path, "w", newline="")
        self._manifest_writer = csv.writer(self._manifest_file)
        self._manifest_writer.writerow(
            ["origin", "embeddings", "target", "split"] + self._metadata_keys
        )

    def update_manifest(
        self,
        input_name: str,
        save_name: str,
        target: str,
        split: str | None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Adds a new entry to the manifest file."""
        metadata_entries = [metadata[key] for key in self._metadata_keys] if metadata else []
        self._manifest_writer.writerow([input_name, save_name, target, split] + metadata_entries)

    def close(self) -> None:
        """Closes the manifest file."""
        if self._manifest_file:
            self._manifest_file.close()
