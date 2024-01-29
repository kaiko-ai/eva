"""PatchCamelyon dataset."""
import os
from typing import Any, Callable, Dict, Literal, Tuple

import h5py
import numpy as np
from typing_extensions import override

from eva.vision.data.datasets import vision

Transform = Callable[..., Any]
""""""


class PatchCamelyon(vision.VisionDataset[Tuple[np.ndarray, np.ndarray]]):
    """"""

    default_column_mapping: Dict[str, str] = {
        "filename": "filename",
        "target": "target",
    }
    """The default column mapping of the variables to the H5 columns."""

    TEST_FILES = [
        "https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz?download=1",
        "https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz?download=1",
        "https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_meta.csv?download=1",
    ]

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "",
        download: bool = False,
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

        self._images = None
        self._targets = None

    @override
    def prepare_data(self) -> None:
        pass

    @override
    def setup(self) -> None:
        if self._split == "train":
            self._h5_data_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_train_x.h5.gz"
            )
            self._h5_targets_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_train_y.h5.gz"
            )
            self._metadata_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_train_meta.csv"
            )
        elif self._split == "val":
            self._h5_data_file = os.path.join(self._root, "camelyonpatch_level_2_split_valid_x.h5")
            self._h5_targets_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_valid_y.h5"
            )
            self._metadata_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_valid_meta.csv"
            )
        elif self._split == "test":
            self._h5_data_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_test_x.h5.gz"
            )
            self._h5_targets_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_test_y.h5.gz"
            )
            self._metadata_file = os.path.join(
                self._root, "camelyonpatch_level_2_split_test_meta.csv"
            )
        else:
            raise ValueError("")

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self._load_image(index)
        # target = self._load_target(index)
        # return self._apply_transforms(image, target)

    @override
    def __len__(self) -> int:
        with h5py.File(self._h5_data_file, "r") as file:
            return len(file[self._column_mapping["filename"]])

    def _load_image(self, index: int) -> np.ndarray:
        """Returns the `index`'th image sample.

        Args:
            index: The index of the data-sample to load.

        Returns:
            The image as a numpy array.
        """
        with h5py.File(self._h5_data_file, "r") as file:
            images = file[self._column_mapping["filename"]]
            return images[self._indices[index] if self._indices else index]

    def _load_target(self, index: int) -> np.ndarray:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data-sample to load.

        Returns:
            The sample target.
        """
        with h5py.File(self._h5_targets_file, "r") as file:
            targets = file[self._column_mapping["target"]]
            target = targets[self._indices[index] if self._indices else index]
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

        if self._image_target_transforms is not None:
            image, target = self._image_target_transforms(image, target)

        return image, target
