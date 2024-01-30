"""Embedding dataset for classification tasks."""

from typing import Dict, Tuple

import pandas as pd
import torch
from typing_extensions import override

from eva.vision.data.datasets.embeddings.embedding import EmbeddingDataset
from eva.vision.data.datasets.typings import DatasetType


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
        dataset_type: DatasetType = DatasetType.PATCH,
        n_patches_per_slide: int = 1000,
        seed: int = 42,
    ):
        """Initialize dataset.

        See docstring of EmbeddingDataset for more details. The only difference is that
        this dataset returns a tuple of (embedding, target) instead of just the embedding.
        """
        super().__init__(
            manifest_path=manifest_path,
            root_dir=root_dir,
            split=split,
            column_mapping=column_mapping,
            dataset_type=dataset_type,
            n_patches_per_slide=n_patches_per_slide,
            seed=seed,
        )

        self._data: pd.DataFrame

        self._target_column = self._column_mapping["target"]

    @override
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: return mask
        return (
            self._data.at[index, self._embedding_column],
            self._data.at[index, self._target_column],
        )
