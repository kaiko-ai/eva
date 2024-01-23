"""Dataset preprocessor base class."""
import abc
import os


class DatasetPreprocessor:
    """Base class for dataset preprocessors."""

    def __init__(self, dataset_dir: str, processed_dir: str) -> None:
        """Initialize the dataset preprocessor.

        Args:
            dataset_dir: Path to the raw dataset directory.
            processed_dir: Path to the output directory where the processed dataset files
                are be stored by the preprocessor.
        """
        self._dataset_dir = dataset_dir
        self._processed_dir = processed_dir

        self._labels_file = os.path.join(processed_dir, "labels.parquet")
        self._splits_file = os.path.join(processed_dir, "splits.parquet")
        self._metadata_file = os.path.join(processed_dir, "metadata.parquet")

    def apply(self):
        """Downloads & preprocesses the dataset into eva's standardized parquet format."""
        self._download_dataset()

        os.makedirs(self._processed_dir, exist_ok=True)
        self._generate_labels_file()
        self._generate_splits_file()
        self._generate_metadata_file()

    @abc.abstractmethod
    def _download_dataset(self):
        """Downloads the dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_labels_file(self):
        """Generates the labels .parquet file with columns: path, label."""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_splits_file(self):
        """Generates the splits .parquet file with columns: path, split."""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_metadata_file(self):
        """Generates the metadata .parquet file with columns: path, ... (other metadata)."""
        raise NotImplementedError
