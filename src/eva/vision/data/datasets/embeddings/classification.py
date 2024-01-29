"""Embedding dataset for classification tasks."""
from typing import Dict, Tuple

import pandas as pd
import torch
from typing_extensions import override

from eva.vision.data.datasets.embeddings.embedding import EmbeddingDataset, default_column_mapping


class EmbeddingClassificationDataset(EmbeddingDataset):
    """Embedding classification dataset."""

    def __init__(
        self,
        manifest_path: str,
        root_dir: str,
        split: str | None,
        column_mapping: Dict[str, str] = default_column_mapping,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to the manifest file.
            root_dir: Root directory of the dataset. If specified, the paths in the manifest
                file are expected to be relative to this directory.
            split: Dataset split to use. If None, the entire dataset is used.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
        """
        super().__init__(
            manifest_path=manifest_path,
            root_dir=root_dir,
            split=split,
            column_mapping=column_mapping,
        )

        self._data: pd.DataFrame

    @override
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:  # pyright: ignore
        return self._data.at[index, self._embedding_column], self._load_target(index)

    def _load_target(self, index) -> torch.Tensor:
        return torch.tensor(self._data.at[index][self._column_mapping["target"]])
