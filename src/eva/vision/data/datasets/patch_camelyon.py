"""PatchCamelyon dataset."""
import os
from typing import Dict, Literal, Tuple

import h5py
import numpy as np
from typing_extensions import override

from eva.data import Transform
from eva.vision.data.datasets import vision


class PatchCamelyon(vision.VisionDataset[Tuple[np.ndarray, np.ndarray]]):
    """Dataset class for PatchCamelyon images and corresponding targets."""

    default_column_mapping: Dict[str, str] = {
        "data": "x",
        "targets": "y",
    }
    """The default column mapping of the variables to the H5 columns."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test"],
        column_mapping: Dict[str, str] = default_column_mapping,
        image_transforms: Transform | None = None,
        target_transforms: Transform | None = None,
        image_target_transforms: Transform | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: The path to the dataset root. This path should contain
                the raw compressed h5 files of the data (`.h5.gz`) and
                the metadata (`.csv`).
            split: The dataset split for training, validation, or testing.
            download: Whether to download the selected dataset split.
            column_mapping: Defines the map between the variables and the CSV
                columns. It will overwrite the `default_column_mapping` with
                the values of `column_mapping`, so that `column_mapping` can
                contain only the values which are altered or missing.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
            image_target_transforms: A function/transforms that takes in an
                image and a label and returns the transformed versions of both.
                This transform happens after the `image_transforms` and
                `target_transforms`.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._column_mapping = self.default_column_mapping | column_mapping
        self._image_transforms = image_transforms
        self._target_transforms = target_transforms
        self._image_target_transforms = image_target_transforms

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self._load_image(index)
        target = self._load_target(index)
        return self._apply_transforms(image, target)

    @override
    def __len__(self) -> int:
        with h5py.File(self._h5_data_file, "r") as file:
            data = file[self._column_mapping["data"]]
            return len(data)

    def _load_image(self, index: int) -> np.ndarray:
        """Returns the `index`'th image sample.

        Args:
            index: The index of the data-sample to load.

        Returns:
            The image as a numpy array.
        """
        with h5py.File(self._h5_data_file, "r") as file:
            data = file.get(self._column_mapping["data"])
            return data[index]

    def _load_target(self, index: int) -> np.ndarray:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data-sample to load.

        Returns:
            The sample target.
        """
        with h5py.File(self._h5_target_file, "r") as file:
            data = file.get(self._column_mapping["targets"])
            return data[index].squeeze()

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

        if self._image_target_transforms is not None:
            image, target = self._image_target_transforms(image, target)

        return image, target

    @property
    def _h5_data_file(self) -> str:
        """Returns the filename for the data H5 file based on the specified data split."""
        return self._h5_file("x")

    @property
    def _h5_target_file(self) -> str:
        """Returns the filename for the target H5 file based on the specified data split."""
        return self._h5_file("y")

    def _h5_file(self, data: Literal["x", "y"]) -> str:
        """Generates the filename for the H5 file based on the specified data type and split.

        Args:
            data: The type of data, where "x" and "y" represent the input and target
                datasets respectively.

        Returns:
            The relative file path for the H5 file based on the provided data type and split.
        """
        filename = f"camelyonpatch_level_2_split_{self._split}_{data}.h5"
        return os.path.join(self._root, filename)
