"""CRC-HE dataset class."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from torchvision.datasets import folder, utils
from typing_extensions import override

from eva.vision.data.datasets import structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io

_URL_TEMPLATE = "https://zenodo.org/records/1214456/files/{filename}.zip?download=1"


class CRC_HE(base.ImageClassification):
    """Dataset class for CRC-HE images and corresponding targets."""

    _n_train_samples = 100000

    _train_resource: structs.DownloadResource = structs.DownloadResource(
        filename="NCT-CRC-HE-100K-NONORM.zip",
        url=_URL_TEMPLATE.format(filename="NCT-CRC-HE-100K-NONORM"),
        md5="md5:035777cf327776a71a05c95da6d6325f",
    )
    """Train resource."""

    _val_resource: structs.DownloadResource = structs.DownloadResource(
        filename="CRC-VAL-HE-7K.zip",
        url=_URL_TEMPLATE.format(filename="CRC-VAL-HE-7K"),
        md5="md5:2fd1651b4f94ebd818ebf90ad2b6ce06",
    )
    """Val resource."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None = None,
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        The CRC-HE dataset is split into a train and validation set:

        - train: 100,000 patches of NCT-CRC-HE-100K-NONORM
        - val: 7,180 patches of CRC-VAL-HE-7K

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.
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

    @property
    def train_dataset_path(self) -> str:
        """Returns the path of train image dataset."""
        return os.path.join(self._root, "NCT-CRC-HE-100K-NONORM")

    @property
    def val_dataset_path(self) -> str:
        """Returns the path of val image dataset."""
        return os.path.join(self._root, "CRC-VAL-HE-7K")

    @override
    def filename(self, index: int) -> str:
        if self._split is None:
            dataset_path = (
                self.train_dataset_path if index < self._n_train_samples else self.val_dataset_path
            )
        elif self._split == "train":
            dataset_path = self.train_dataset_path
        else:
            dataset_path = self.val_dataset_path
        image_path, _ = self._samples[index]
        return os.path.relpath(image_path, dataset_path)

    @override
    def prepare_data(self) -> None:
        if self._download and self._split in ["train", None]:
            if not os.path.isdir(self.train_dataset_path):
                self._download_dataset(self._train_resource)
        if self._download and self._split in ["val", None]:
            if not os.path.isdir(self.val_dataset_path):
                self._download_dataset(self._val_resource)

    @override
    def setup(self) -> None:
        samples = []
        if self._split in ["train", None]:
            samples += self._make_dataset(self.train_dataset_path)
        if self._split in ["val", None]:
            samples += self._make_dataset(self.val_dataset_path)
        self._samples = samples

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

    def _make_dataset(self, directory: str) -> List[Tuple[str, int]]:
        """Builds the dataset from the specified directory."""
        return folder.make_dataset(
            directory=directory,
            class_to_idx=self.class_to_idx,
            extensions=(".tif"),
        )

    def _download_dataset(self, resource: structs.DownloadResource) -> None:
        """Downloads the CRC HE datasets."""
        utils.download_and_extract_archive(
            resource.url,
            download_root=self._root,
            filename=resource.filename,
            remove_finished=True,
        )
