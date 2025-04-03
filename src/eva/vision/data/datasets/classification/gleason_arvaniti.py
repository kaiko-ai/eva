"""GleasonArvaniti dataset class."""

import functools
import glob
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal

import pandas as pd
import torch
from loguru import logger
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision
from eva.vision.utils import io


class GleasonArvaniti(vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for GleasonArvaniti images and corresponding targets."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 15303,
        "val": 2482,
        "test": 4967,
        None: 22752,
    }
    """Expected dataset lengths for the splits and complete dataset."""

    _license: str = "CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Path to the root directory of the dataset.
            split: Dataset split to use. If `None`, the entire dataset is used.
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split

        self._indices: List[int] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["benign", "gleason_3", "gleason_4", "gleason_5"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.classes)}

    @functools.cached_property
    def _image_files(self) -> List[str]:
        """Return the list of image files in the dataset.

        Returns:
            List of image file paths.
        """
        subdirs = ["train_validation_patches_750", "test_patches_750/patho_1"]

        image_files = []
        for subdir in subdirs:
            files_pattern = os.path.join(self._root, subdir, "**/*.jpg")
            image_files += list(glob.glob(files_pattern, recursive=True))

        return sorted(image_files)

    @functools.cached_property
    def _manifest(self) -> pd.DataFrame:
        """Returns the train.csv & test.csv files as dataframe."""
        df_train = pd.read_csv(os.path.join(self._root, "train.csv"))
        df_val = pd.read_csv(os.path.join(self._root, "test.csv"))
        df_train["split"], df_val["split"] = "train", "val"
        return pd.concat([df_train, df_val], axis=0).set_index("image_id")

    @override
    def filename(self, index: int) -> str:
        image_path = self._image_files[self._indices[index]]
        return os.path.relpath(image_path, self._root)

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, download_available=False)
        if not os.path.isdir(os.path.join(self._root, "train_validation_patches_750")):
            raise FileNotFoundError(
                f"`train_validation_patches_750` directory not found in {self._root}"
            )
        if not os.path.isdir(os.path.join(self._root, "test_patches_750")):
            raise FileNotFoundError(f"`test_patches_750` directory not found in {self._root}")

        if self._split == "test":
            logger.warning(
                "The test split currently leads to unstable evaluation results. "
                "We recommend using the validation split instead."
            )

    @override
    def configure(self) -> None:
        self._indices = self._make_indices()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths[self._split],
            n_classes=4,
            first_and_last_labels=("benign", "gleason_5"),
        )

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_path = self._image_files[self._indices[index]]
        return io.read_image_as_tensor(image_path)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        target = self._extract_class(self._image_files[self._indices[index]])
        return torch.tensor(target, dtype=torch.long)

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")

    def _extract_micro_array_id(self, file: str) -> str:
        """Extracts the ID of the tissue micro array from the file name."""
        return Path(file).stem.split("_")[0]

    def _extract_class(self, file: str) -> int:
        """Extracts the class label from the file name."""
        return int(Path(file).stem.split("_")[-1])

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        train_indices, val_indices, test_indices = [], [], []

        for index, image_file in enumerate(self._image_files):
            array_id = self._extract_micro_array_id(image_file)

            if array_id == "ZT76":
                val_indices.append(index)
            elif array_id in {"ZT111", "ZT199", "ZT204"}:
                train_indices.append(index)
            elif "test_patches_750" in image_file:
                test_indices.append(index)
            else:
                raise ValueError(f"Invalid microarray value found for file {image_file}")

        split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            None: train_indices + val_indices + test_indices,
        }

        return split_indices[self._split]
