"""Dataset class for patch embeddings."""

import os
from typing import Dict, Literal, Tuple

import pandas as pd
import torch
from typing_extensions import override

from eva.vision.data.datasets.vision import VisionDataset


class PatchEmbeddingDataset(VisionDataset):
    """Embedding dataset."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
        "split": "split",
    }

    def __init__(
        self,
        root: str,
        manifest_path: str | None,
        split: Literal["train", "val", "test"],
        column_mapping: Dict[str, str] | None = None,
    ):
        """Initialize dataset.

        Expects a manifest file listing the paths of .pt files that contain tensor embeddings
        of shape [embedding_dim] or [1, embedding_dim].

        Args:
            root: Root directory of the dataset. If specified, the paths in the manifest
                file are expected to be relative to this directory.
            manifest_path: Path to the manifest file. Can be either a .csv or .parquet file, with
                the required columns: path, target, split (names can be adjusted using the
                column_mapping parameter). If None, will default to `<root>/manifest.csv`.
            split: Dataset split to use.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._manifest_path = manifest_path if manifest_path else os.path.join(root, "manifest.csv")

        self._data: pd.DataFrame

        self._column_mapping = self.default_column_mapping
        if column_mapping:
            self._column_mapping.update(column_mapping)
        self._path_column = self._column_mapping["path"]
        self._target_column = self._column_mapping["target"]
        self._split_column = self._column_mapping["split"]

    @override
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self._load_embedding_file(index)
        target = self._data.at[index, self._target_column]
        return embedding, target

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def filename(self, index: int) -> str:
        return self._data.at[index, self._path_column]

    @override
    def setup(self):
        self._data = self._load_manifest()
        self._data = self._data.loc[self._data[self._split_column] == self._split]
        self._data = self._data.reset_index(drop=True)

    def _load_embedding_file(self, index) -> torch.Tensor:
        path = self._get_embedding_path(index)
        tensor = torch.load(path, map_location="cpu").squeeze(0)
        if tensor.ndim != 1:
            raise ValueError(f"Unexpected tensor shape {tensor.shape} for {path}")
        return tensor

    def _get_embedding_path(self, index: int) -> str:
        return os.path.join(self._root, self.filename(index))

    def _load_manifest(self) -> pd.DataFrame:
        if self._manifest_path.endswith(".csv"):
            return pd.read_csv(self._manifest_path)
        elif self._manifest_path.endswith(".parquet"):
            return pd.read_parquet(self._manifest_path)
        else:
            raise ValueError(f"Failed to load manifest file {self._manifest_path}")
