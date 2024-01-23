"""Core Dataset module."""
from typing import Tuple, Type

import pandas as pd
import torch
from typing_extensions import override

from eva.data.preprocessors import DatasetPreprocessor
from eva.vision.data.datasets.embeddings.embedding import EmbeddingDataset


class EmbeddingClassificationDataset(EmbeddingDataset):
    """Embedding classification dataset."""

    def __init__(
        self,
        dataset_dir: str,
        preprocessor: Type[DatasetPreprocessor],
        processed_dir: str,
        path_mappings_file: str | None,
        split: str | None,
    ):
        """Initialize dataset.

        Args:
            dataset_dir: Path to the dataset directory.
            preprocessor: Dataset preprocessor.
            processed_dir: Path to the output directory where the processed dataset files
                are be stored by the preprocessor.
            path_mappings_file: Path to the file containing the mappings between the original
                image paths and the corresponding embedding paths. If not specified, the
                paths will not be mapped.
            split: Dataset split to use. If None, the entire dataset is used.
        """
        super().__init__(dataset_dir, preprocessor, processed_dir, path_mappings_file, split)

        self._preprocessor = preprocessor(dataset_dir, processed_dir)

        self._data: pd.DataFrame

    def _load_target(self, index) -> torch.Tensor:
        return torch.tensor(self._data.at[index][self._column_mapping["target"]])

    @override
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:  # pyright: ignore
        """Get a sample from the dataset."""
        embedding, target = self._load_embedding(index), self._load_target(index)
        return embedding, target
