"""Core Dataset module."""
from typing import Dict

import pandas as pd
import torch
from typing_extensions import override

from eva.vision.data.datasets.vision import VisionDataset

default_column_mapping: Dict[str, str] = {
    "path": "path",
    "target": "target",
}


class EmbeddingDataset(VisionDataset):
    """Embedding dataset."""

    def __init__(
        self,
        manifest_path: str,
        split: str | None,
        column_mapping: Dict[str, str] = default_column_mapping,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to the manifest file.
            split: Dataset split to use. If None, the entire dataset is used.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
        """
        super().__init__()

        self._manifest_path = manifest_path
        self._split = split
        self._column_mapping = column_mapping

        self._data: pd.DataFrame

    def _load_embedding(self, index) -> torch.Tensor:
        return torch.load(self._data.at[index][self._column_mapping["path"]], map_location="cpu")

    @override
    def __getitem__(self, index) -> torch.Tensor:
        """Get a sample from the dataset."""
        return self._load_embedding(index)

    def setup(self):
        """Setup dataset."""
        self._data = self._load_manifest()
        self._aggregate_embeddings()

    def _load_manifest(self) -> pd.DataFrame:
        """Load manifest file."""
        return pd.read_parquet(self._manifest_path)

    def _aggregate_embeddings(self) -> pd.DataFrame:
        """Function to aggregate embeddings."""
        raise NotImplementedError
