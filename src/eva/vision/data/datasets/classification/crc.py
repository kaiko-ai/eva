"""CRC dataset class."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from torchvision.datasets import folder, utils
from typing_extensions import override

from eva.vision.data.datasets import _validators, structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class CRC(base.ImageClassification):
    """Dataset class for CRC images and corresponding targets."""

    _train_resource: structs.DownloadResource = structs.DownloadResource(
        filename="NCT-CRC-HE-100K.zip",
        url="https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1",
        md5="md5:035777cf327776a71a05c95da6d6325f",
    )
    """Train resource."""

    _val_resource: structs.DownloadResource = structs.DownloadResource(
        filename="CRC-VAL-HE-7K.zip",
        url="https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1",
        md5="md5:2fd1651b4f94ebd818ebf90ad2b6ce06",
    )
    """Validation resource."""

    _license: str = "CC BY 4.0 LEGAL CODE (https://creativecommons.org/licenses/by/4.0/legalcode)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        The dataset is split into a train (train) and validation (val) set:
          - train: A set of 100,000 non-overlapping image patches from
            hematoxylin & eosin (H&E) stained histological images of human
            colorectal cancer (CRC) and normal tissue.
          - val: A set of 7180 image patches from N=50 patients with colorectal
            adenocarcinoma (no overlap with patients in NCT-CRC-HE-100K).

        Args:
            root: Path to the root directory of the dataset.
            split: Dataset split to use.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does
                not yet exist on disk.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
        """
        super().__init__(
            image_transforms=image_transforms,
            target_transforms=target_transforms,
        )

        self._root = root
        self._split = split
        self._download = download

        self._samples: List[Tuple[str, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {
            "ADI": 0,
            "BACK": 1,
            "DEB": 2,
            "LYM": 3,
            "MUC": 4,
            "MUS": 5,
            "NORM": 6,
            "STR": 7,
            "TUM": 8,
        }

    @override
    def filename(self, index: int) -> str:
        image_path, *_ = self._samples[index]
        return os.path.relpath(image_path, self._dataset_dir)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def configure(self) -> None:
        self._samples = self._make_dataset()

    @override
    def validate(self) -> None:
        expected_length = {
            "train": 100000,
            "val": 7180,
            None: 107180,
        }
        _validators.check_dataset_integrity(
            self,
            length=expected_length.get(self._split, 0),
            n_classes=9,
            first_and_last_labels=("ADI", "TUM"),
        )

    @override
    def load_image(self, index: int) -> np.ndarray:
        image_path, _ = self._samples[index]
        return io.read_image(image_path)

    @override
    def load_target(self, index: int) -> np.ndarray:
        _, target = self._samples[index]
        return np.asarray(target, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return len(self._samples)

    @property
    def _dataset_dir(self) -> str:
        """Returns the full path of dataset directory."""
        dataset_dirs = {
            "train": os.path.join(self._root, "NCT-CRC-HE-100K"),
            "val": os.path.join(self._root, "CRC-VAL-HE-7K"),
        }
        dataset_dir = dataset_dirs.get(self._split)
        if dataset_dir is None:
            raise ValueError("Invalid data split. Use 'train' or 'val'.")

        return dataset_dir

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Builds the dataset for the specified split."""
        dataset = folder.make_dataset(
            directory=self._dataset_dir,
            class_to_idx=self.class_to_idx,
            extensions=(".tif"),
        )
        return dataset

    def _download_dataset(self) -> None:
        """Downloads the dataset resources."""
        for resource in [self._train_resource, self._val_resource]:
            resource_dir = resource.filename.rsplit(".", maxsplit=1)[0]
            if os.path.isdir(os.path.join(self._root, resource_dir)):
                continue

            self._print_license()
            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
