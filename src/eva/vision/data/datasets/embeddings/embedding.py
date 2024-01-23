"""Core Dataset module."""
from typing import Type

import pandas as pd
import torch

from eva.data.preprocessors import DatasetPreprocessor
from eva.vision.data.datasets.vision import VisionDataset


class EmbeddingDataset(VisionDataset[torch.Tensor]):
    """Embedding dataset."""

    def __init__(
        self,
        dataset_dir: str,
        preprocessor: Type[DatasetPreprocessor],
        processed_dir: str,
        path_mappings_file: str | None,
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
        super().__init__(dataset_dir, preprocessor, processed_dir)

        self._path_mappings_file = path_mappings_file
        self._preprocessor = preprocessor(dataset_dir, processed_dir)

        self._data: pd.DataFrame

    def _load_embedding(self, index) -> torch.Tensor:
        return torch.load(self._data.at[index][self._column_mapping["path"]], map_location="cpu")

    def __getitem__(self, index) -> torch.Tensor:
        """Get a sample from the dataset."""
        return self._load_embedding(index)

    def setup(self):
        """Setup dataset."""
        super().setup()
        self._map_paths()
        self._aggregate_embeddings()

    def _map_paths(self) -> pd.DataFrame:
        """Function to map the image paths to the corresponding embedding paths."""
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError("Call setup() first")
        if not self._path_mappings_file:
            return self._data
        df_mappings = pd.read_parquet(self._path_mappings_file)
        df = pd.merge(
            self._data,
            df_mappings,
            on=self._column_mapping["path"],
            how="left",
            validate="one_to_many",
        )

        df = df.drop(self._column_mapping["path"], axis=1).rename(
            {"map_to": self._column_mapping["path"]}, axis=1
        )

        return df

    def _aggregate_embeddings(self) -> pd.DataFrame:
        """Function to aggregate embeddings."""
        raise NotImplementedError
