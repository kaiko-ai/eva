"""CRC-HE dataset class."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from torchvision.datasets import folder, utils
from typing_extensions import override

from eva.vision.data.datasets import _utils, structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io

_URL_TEMPLATE = "https://zenodo.org/records/1214456/files/{filename}.zip?download=1"


class CRC_HE(base.ImageClassification):
    """Dataset class for CRC-HE images and corresponding targets."""

    _train_index_ranges: List[Tuple[int, int]] = [
        (0, 8326),
        (10407, 18860),
        (20973, 30183),
        (32485, 41731),
        (44042, 51159),
        (52938, 63767),
        (66474, 73485),
        (75237, 83594),
        (85683, 97137),
    ]
    """Train range indices."""

    _val_index_ranges: List[Tuple[int, int]] = [
        (8326, 10407),
        (18860, 20973),
        (30183, 32485),
        (41731, 44042),
        (51159, 52938),
        (63767, 66474),
        (73485, 75237),
        (83594, 85683),
        (97137, 100000),
    ]
    """Validation range indices."""

    _test_index_ranges: List[Tuple[int, int]] = [
        (100000, 107180),
    ]
    """Test range indices."""

    _train_val_resource: structs.DownloadResource = structs.DownloadResource(
        filename="NCT-CRC-HE-100K-NONORM.zip",
        url=_URL_TEMPLATE.format(filename="NCT-CRC-HE-100K-NONORM"),
        md5="md5:035777cf327776a71a05c95da6d6325f",
    )
    """Train / Validation resources."""

    _test_resource: structs.DownloadResource = structs.DownloadResource(
        filename="CRC-VAL-HE-7K.zip",
        url=_URL_TEMPLATE.format(filename="CRC-VAL-HE-7K"),
        md5="md5:2fd1651b4f94ebd818ebf90ad2b6ce06",
    )
    """Test resources."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None = None,
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        The CRC-HE dataset is split into a train, validation and test set:

        - train: 80% of NCT-CRC-HE-100K-NONORM (stratifed by class)
        - val: 20% of NCT-CRC-HE-100K-NONORM (stratifed by class)
        - test: 100% of CRC-VAL-HE-7K

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
        self._indices: List[int] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["Benign", "InSitu", "Invasive", "Normal"]

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
    def train_val_dataset_path(self) -> str:
        """Returns the path of train & val image data of the dataset."""
        return os.path.join(self._root, "NCT-CRC-HE-100K-NONORM")

    @property
    def test_dataset_path(self) -> str:
        """Returns the path of test image data of the dataset."""
        return os.path.join(self._root, "CRC-VAL-HE-7K")

    @override
    def filename(self, index: int) -> str:
        image_path, _ = self._samples[self._indices[index]]
        if self.train_val_dataset_path in image_path:
            return os.path.relpath(image_path, self.train_val_dataset_path)
        else:
            return os.path.relpath(image_path, self.test_dataset_path)

    @override
    def prepare_data(self) -> None:
        if self._download:
            if not os.path.isdir(self.train_val_dataset_path):
                self._download_dataset(self._train_val_resource)
            if not os.path.isdir(self.test_dataset_path):
                self._download_dataset(self._test_resource)

    @override
    def setup(self) -> None:
        train_val_samples = self._make_dataset(self.train_val_dataset_path)
        test_samples = self._make_dataset(self.test_dataset_path)
        self._samples = train_val_samples + test_samples
        self._indices = self._make_indices()

    @override
    def load_image(self, index: int) -> np.ndarray:
        image_path, _ = self._samples[self._indices[index]]
        return io.read_image(image_path)

    @override
    def load_target(self, index: int) -> np.ndarray:
        _, target = self._samples[self._indices[index]]
        return np.asarray(target, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return len(self._indices)

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

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        split_index_ranges = {
            "train": self._train_index_ranges,
            "val": self._val_index_ranges,
            "test": self._test_index_ranges,
            None: [(0, 107180)],
        }
        index_ranges = split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val', 'test' or `None`.")

        return _utils.ranges_to_indices(index_ranges)
