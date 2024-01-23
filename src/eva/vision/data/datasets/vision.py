"""Core Dataset module."""
from typing import Dict, Type

import pandas as pd

from eva.data.datasets.dataset import Dataset
from eva.data.preprocessors import DatasetPreprocessor


class VisionDataset(Dataset):
    """Vision base dataset class.

    For all benchmark datasets that use eva's standardized parquet format
    for storing labels, splits and metadata.
    """

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "label",
    }

    def __init__(
        self,
        dataset_dir: str,
        preprocessor: Type[DatasetPreprocessor],
        processed_dir: str,
        split: str | None,
        column_mapping: Dict[str, str] = default_column_mapping,
    ):
        """Initialize dataset.

        Args:
            dataset_dir: Path to the dataset directory.
            preprocessor: Dataset preprocessor.
            processed_dir: Path to the output directory where the processed dataset files
                are be stored by the preprocessor.
            split: Dataset split to use. If None, the entire dataset is used.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the dataset parquet files.
        """
        super().__init__()

        self._dataset_dir = dataset_dir
        self._processed_dir = processed_dir
        self._preprocessor = preprocessor(dataset_dir, processed_dir)
        self._split = split
        self._column_mapping = column_mapping

        self._data: pd.DataFrame

    def setup(self):
        """Setup dataset."""
        self._preprocessor.apply()
        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Loads the labels, splits and metadata files and merges them into a single dataframe."""
        df_labels = pd.read_parquet(self._preprocessor._labels_file)
        df_splits = pd.read_parquet(self._preprocessor._splits_file)

        data = pd.merge(
            df_labels, df_splits, on=self._column_mapping["path"], validate="one_to_one"
        )

        if self._preprocessor._metadata_file:
            df_metadata = pd.read_parquet(self._preprocessor._metadata_file)
            data = pd.merge(data, df_metadata, on=self._column_mapping["path"])

        if self._split:
            data = data.loc[data["split"] == self._split]

        return data
