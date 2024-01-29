"""PatchCamelyon dataset."""

import os
from typing import Callable, Dict, Literal, Tuple

import h5py
import numpy as np
from typing_extensions import override

from eva.vision.data.datasets import vision


class PatchCamelyon(vision.VisionDataset[Tuple[np.ndarray, np.ndarray]]):
    """Dataset class for PatchCamelyon images and corresponding targets."""

    column_mapping: Dict[str, Literal["x", "y"]] = {
        "data": "x",
        "targets": "y",
    }
    """The column mapping of the variables to the H5 columns."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test"],
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: The path to the dataset root. This path should contain
                the raw compressed h5 files of the data (`.h5.gz`) and
                the metadata (`.csv`).
            split: The dataset split for training, validation, or testing.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._image_transforms = image_transforms
        self._target_transforms = target_transforms

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self._load_image(index)
        target = self._load_target(index)
        return self._apply_transforms(image, target)

    @override
    def __len__(self) -> int:
        return self._load_from_h5("targets").shape[0]

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
        return self._load_from_h5("targets", index).squeeze()

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
            return data if index is None else data[index]  # type: ignore

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
