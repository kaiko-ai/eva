"""Core Dataset module."""
from typing import Optional, Tuple, Type, TypeVar

import numpy as np
import pandas as pd

from eva.data.preprocessors import DatasetPreprocessor
from eva.vision.data.datasets.embeddings.embedding import EmbeddingDataset

DataSample = TypeVar("DataSample")


class EmbeddingClassificationDataset(EmbeddingDataset[Tuple[np.ndarray, np.ndarray]]):
    """Embedding classification dataset."""

    def __init__(
        self,
        dataset_dir: str,
        preprocessor: Type[DatasetPreprocessor],
        processed_dir: str,
        path_mappings_file: Optional[str],
        **kwargs,
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
        """
        super().__init__(dataset_dir, preprocessor, processed_dir, path_mappings_file, **kwargs)
        self._preprocessor = preprocessor(dataset_dir, processed_dir)
        self._data: pd.DataFrame

    def _load_target(self, index) -> np.ndarray:
        return self._data.at[index][self._column_mapping["target"]]

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        """Get a sample from the dataset."""
        embedding, target = self._load_embedding(index), self._load_target(index)
        return embedding, target
