"""UniToPatho dataset class."""

import functools
import glob
import os
from typing import Callable, Dict, List, Literal

import pandas as pd
import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision
from eva.vision.utils import io


class UniToPatho(vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for UniToPatho images and corresponding targets."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 6270,
        "val": 2399,
        None: 8669,
    }
    """Expected dataset lengths for the splits and complete dataset."""

    _license: str = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        The dataset is split into train and validation by taking into account
        the patient IDs to avoid any data leakage.

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
        return ["HP", "NORM", "TA.HG", "TA.LG", "TVA.HG", "TVA.LG"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {"HP": 0, "NORM": 1, "TA.HG": 2, "TA.LG": 3, "TVA.HG": 4, "TVA.LG": 5}

    @property
    def _dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        return os.path.join(self._root, "800")

    @functools.cached_property
    def _image_files(self) -> List[str]:
        """Return the list of image files in the dataset.

        Returns:
            List of image file paths.
        """
        files_pattern = os.path.join(self._dataset_path, "**/*.png")
        image_files = list(glob.glob(files_pattern, recursive=True))
        return sorted(image_files)

    @functools.cached_property
    def _manifest(self) -> pd.DataFrame:
        """Returns the train.csv & test.csv files as dataframe."""
        df_train = pd.read_csv(os.path.join(self._dataset_path, "train.csv"))
        df_val = pd.read_csv(os.path.join(self._dataset_path, "test.csv"))
        df_train["split"], df_val["split"] = "train", "val"
        return pd.concat([df_train, df_val], axis=0).set_index("image_id")

    @override
    def filename(self, index: int) -> str:
        image_path = self._image_files[self._indices[index]]
        return os.path.relpath(image_path, self._dataset_path)

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, True)

    @override
    def configure(self) -> None:
        self._indices = self._make_indices()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths[self._split],
            n_classes=6,
            first_and_last_labels=("HP", "TVA.LG"),
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

    def _extract_image_id(self, image_file: str) -> str:
        """Extracts the image_id from the file name."""
        return os.path.basename(image_file)

    def _extract_class(self, file: str) -> int:
        image_id = self._extract_image_id(file)
        return int(self._manifest.at[image_id, "top_label"])

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        train_indices = []
        val_indices = []

        for index, image_file in enumerate(self._image_files):
            image_id = self._extract_image_id(image_file)
            split = self._manifest.at[image_id, "split"]

            if split == "train":
                train_indices.append(index)
            elif split == "val":
                val_indices.append(index)
            else:
                raise ValueError(f"Invalid split value found: {split}")

        split_indices = {
            "train": train_indices,
            "val": val_indices,
            None: train_indices + val_indices,
        }

        return split_indices[self._split]
