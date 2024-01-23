"""Bach data preprocessing."""
from typing_extensions import override

from eva.data.preprocessors import DatasetPreprocessor


class BachPreprocessor(DatasetPreprocessor):
    """Dataset preprocessor for Bach dataset."""

    def __init__(self, dataset_dir: str, processed_dir: str) -> None:
        """Dataset preprocessor for Bach dataset.

        Args:
            dataset_dir: Path to the raw dataset directory.
            processed_dir: Path to the output directory where the processed dataset files
                are be stored by the preprocessor.
        """
        super().__init__(dataset_dir, processed_dir)

    @override
    def _generate_labels_file(self):
        raise NotImplementedError

    @override
    def _generate_splits_file(self):
        raise NotImplementedError

    @override
    def _generate_metadata_file(self):
        raise NotImplementedError
