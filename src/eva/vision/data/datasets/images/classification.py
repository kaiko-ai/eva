"""Core Dataset module."""
from typing import Tuple, Type

import numpy as np
import pandas as pd
from typing_extensions import override

from eva.data.preprocessors import DatasetPreprocessor
from eva.vision.data.datasets.images.image import ImageDataset


class ImageClassificationDataset(ImageDataset):
    """Image classification dataset."""

    def __init__(
        self,
        dataset_dir: str,
        preprocessor: Type[DatasetPreprocessor],
        processed_dir: str,
        split: str | None,
    ):
        """Initialize dataset.

        Args:
            dataset_dir: Path to the dataset directory.
            preprocessor: Dataset preprocessor.
            processed_dir: Path to the output directory where the processed dataset files
                are be stored by the preprocessor.
            split: Dataset split to use. If None, the entire dataset is used.
        """
        super().__init__(dataset_dir, preprocessor, processed_dir, split)

        self._preprocessor = preprocessor(dataset_dir, processed_dir)

        self._data: pd.DataFrame

    def _load_target(self, index) -> np.ndarray:
        return self._data.at[index][self._column_mapping["target"]]

    @override
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:  # pyright: ignore
        """Get a sample from the dataset."""
        target = np.asarray(self._data.at[index][self._column_mapping["target"]], dtype=np.int64)
        image = self._load_image(index)

        return image, target
