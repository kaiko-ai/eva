"""Embedding dataset."""
import dataclasses
import os
from typing import Dict

import pandas as pd
import torch
import tqdm
from typing_extensions import override

from eva.vision.data.datasets.vision import VisionDataset

default_column_mapping: Dict[str, str] = {
    "path": "path",
    "target": "target",
}


@dataclasses.dataclass
class AggregationConfig:
    """Embedding aggregation configuration for slide level tasks."""

    n_patches: int = 1000
    seed: int = 42


class EmbeddingDataset(VisionDataset):
    """Embedding dataset."""

    def __init__(
        self,
        manifest_path: str,
        root_dir: str,
        split: str | None,
        column_mapping: Dict[str, str] = default_column_mapping,
        aggregation_config: AggregationConfig | None = None,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to the manifest file.
            root_dir: Root directory of the dataset. If specified, the paths in the manifest
                file are expected to be relative to this directory.
            split: Dataset split to use. If None, the entire dataset is used.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
            aggregation_config: Embedding aggregation configuration for slide level tasks.
                If None, no aggregation is performed.
        """
        super().__init__()

        self._manifest_path = manifest_path
        self._root_dir = root_dir
        self._split = split
        self._column_mapping = column_mapping
        self._aggregation_config = aggregation_config

        self._data: pd.DataFrame

        self._path_column = self._column_mapping["path"]
        self._embedding_column = self._column_mapping["embedding"]

    @override
    def __getitem__(self, index) -> torch.Tensor:
        return self._data.at[index, self._embedding_column]

    @override
    def setup(self):
        self._data = self._load_manifest()
        self._data[self._embedding_column] = None

        for index, _ in tqdm.tqdm(self._data):
            self._data.at[index, self._embedding_column] = self._load_embedding_file(index)

        if self._aggregation_config is not None:
            self._aggregate_embeddings()

    def _load_embedding_file(self, index) -> torch.Tensor:
        return torch.load(self._get_embedding_path(index), map_location="cpu")

    def _get_embedding_path(self, index: int) -> str:
        return os.path.join(self._root_dir, self._data.at[index, self._path_column])

    def _load_manifest(self) -> pd.DataFrame:
        return pd.read_parquet(self._manifest_path)

    def _aggregate_embeddings(self) -> pd.DataFrame:
        raise NotImplementedError
