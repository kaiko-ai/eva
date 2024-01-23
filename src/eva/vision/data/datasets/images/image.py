"""Core Dataset module."""
from typing import Type

import cv2
import numpy as np
import pandas as pd
from typing_extensions import override

from eva.data.preprocessors import DatasetPreprocessor
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.file_io import image_io


class ImageDataset(VisionDataset):
    """Image dataset."""

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

    def _load_image(self, index) -> np.ndarray:
        """Load an image from file."""
        image = image_io.load_image_file_as_array(
            self._data.at[index][self._column_mapping["path"]], cv2.IMREAD_COLOR
        )
        return image

    @override
    def __getitem__(self, index) -> np.ndarray:
        """Get a sample from the dataset."""
        return self._load_image(index)
