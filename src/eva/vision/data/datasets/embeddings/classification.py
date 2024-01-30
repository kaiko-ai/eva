"""Embedding dataset for classification tasks."""

from typing import Dict, Tuple

import pandas as pd
import torch
from typing_extensions import override

from eva.vision.data.datasets.embeddings.embedding import EmbeddingDataset


class EmbeddingClassificationDataset(EmbeddingDataset):
    """Embedding classification dataset."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
        "slide_id": "slide_id",
        "mask": "mask",
    }

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

        self._target_column = self._column_mapping["target"]

    @override
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._data.at[index, self._embedding_column],
            self._data.at[index][self._target_column],
        )
