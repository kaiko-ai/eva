"""PatchCamelyon dataset."""

import os
from typing import Callable, Dict, Literal, Tuple

import h5py
import numpy as np
from loguru import logger
from typing_extensions import override

from eva.utils import file_io
from eva.vision.data.datasets import vision


class PatchCamelyon(vision.VisionDataset[Tuple[np.ndarray, np.ndarray]]):
    """Dataset class for PatchCamelyon images and corresponding targets."""

    column_mapping: Dict[str, Literal["x", "y"]] = {
        "data": "x",
        "targets": "y",
    }
    """The column mapping of the variables to the H5 columns."""

    file_list = [
        ("camelyonpatch_level_2_split_train_x.h5", 7248299800),
        ("camelyonpatch_level_2_split_train_y.h5", 264192),
        ("camelyonpatch_level_2_split_valid_x.h5", 906040528),
        ("camelyonpatch_level_2_split_valid_y.h5", 34816),
        ("camelyonpatch_level_2_split_test_x.h5", 906040528),
        ("camelyonpatch_level_2_split_test_y.h5", 34816),
    ]
    """PatchCamelyon dataset file lists."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test"] = "train",
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: The path to the dataset root. This path should contain
                the uncompressed h5 files and the metadata.
            split: The dataset split for training, validation, or testing.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._download = download
        self._image_transforms = image_transforms
        self._target_transforms = target_transforms

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    def _download_dataset(self) -> None:
        """Downloads the PatchCamelyon dataset."""
        for file, expected_bytes in self.file_list:
            filename = os.path.join(self._root, file)
            if file_io.verify_file(filename, expected_bytes):
                logger.info(f"File '{filename}' already exists. Skipping downloading.")
                continue

            logger.info(f"Downloading file '{filename}' has started.")
            url = self._filename_to_url(file)
            file_io.download_and_extract_archive(url, output_dir=self._root)
            logger.info(f"Downloading file '{filename}' finished successfully.")

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self._load_image(index)
        target = self._load_target(index)
        return self._apply_transforms(image, target)

    @override
    def __len__(self) -> int:
        return self._fetch_dataset_length()

    def _load_image(self, index: int) -> np.ndarray:
        """Returns the `index`'th image sample.

        Args:
            index: The index of the data-sample to load.

        Returns:
            The image as a numpy array.
        """
        return self._load_from_h5("data", index)

    def _load_target(self, index: int) -> np.ndarray:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data-sample to load.

        Returns:
            The sample target.
        """
        target = self._load_from_h5("targets", index).squeeze()
        return np.asarray(target, dtype=np.int64)

    def _apply_transforms(
        self, image: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the transforms to the provided data and returns them.

        Args:
            image: The desired image.
            target: The target of the image.

        Returns:
            A tuple with the image and the target transformed.
        """
        if self._image_transforms is not None:
            image = self._image_transforms(image)

        if self._target_transforms is not None:
            target = self._target_transforms(target)

        return image, target

    def _fetch_dataset_length(self) -> int:
        """Fetches the dataset split length from its HDF5 file."""
        data_key = self.column_mapping["targets"]
        h5_file = self._h5_file(data_key)
        with h5py.File(h5_file, "r") as file:
            data = file[data_key]
            return len(data)  # type: ignore

    def _load_from_h5(
        self, datatype: Literal["data", "targets"], index: int | None = None
    ) -> np.ndarray:
        """Load data or targets from an HDF5 file.

        Args:
            datatype: Specify whether to load 'data' or 'targets'.
            index: Optional parameter to load data/targets at a specific index.
                If `None`, the entire data/targets array is returned.

        Returns:
            A array containing the specified data.
        """
        data_key = self.column_mapping[datatype]
        h5_file = self._h5_file(data_key)
        with h5py.File(h5_file, "r") as file:
            data = file[data_key]
            return data[:] if index is None else data[index]  # type: ignore

    def _h5_file(self, datatype: Literal["x", "y"]) -> str:
        """Generates the filename for the H5 file based on the specified data type and split.

        Args:
            datatype: The type of data, where "x" and "y" represent the input
                and target datasets respectively.

        Returns:
            The relative file path for the H5 file based on the provided data type and split.
        """
        filename = f"camelyonpatch_level_2_split_{self._split}_{datatype}.h5"
        return os.path.join(self._root, filename)

    @staticmethod
    def _filename_to_url(filename: str) -> str:
        """Convert a filename to its corresponding download URL."""
        return f"https://zenodo.org/records/2546921/files/{filename}.gz?download=1"
