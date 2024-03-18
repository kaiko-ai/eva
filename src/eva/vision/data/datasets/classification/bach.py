"""BACH dataset class."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from torchvision.datasets import folder, utils
from typing_extensions import override

from eva.vision.data.datasets import _utils, _validators, structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class BACH(base.ImageClassification):
    """Dataset class for BACH images and corresponding targets."""

    _train_index_ranges: List[Tuple[int, int]] = [
        (0, 41),
        (59, 60),
        (90, 139),
        (169, 240),
        (258, 260),
        (273, 345),
        (368, 400),
    ]
    """Train range indices."""

    _val_index_ranges: List[Tuple[int, int]] = [
        (41, 59),
        (60, 90),
        (139, 169),
        (240, 258),
        (260, 273),
        (345, 368),
    ]
    """Validation range indices."""

    _resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="ICIAR2018_BACH_Challenge.zip",
            url="https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip",
        ),
    ]
    """Dataset resources."""

    _license: str = "CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None = None,
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        The dataset is split into train and validation by taking into account
        the patient IDs to avoid any data leakage.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If `None`, the entire dataset is used.
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
        return {"Benign": 0, "InSitu": 1, "Invasive": 2, "Normal": 3}

    @property
    def dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        return os.path.join(self._root, "ICIAR2018_BACH_Challenge", "Photos")

    @override
    def filename(self, index: int) -> str:
        image_path, _ = self._samples[self._indices[index]]
        return os.path.relpath(image_path, self.dataset_path)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def configure(self) -> None:
        self._samples = folder.make_dataset(
            directory=self.dataset_path,
            class_to_idx=self.class_to_idx,
            extensions=(".tif"),
        )
        self._indices = self._make_indices()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=268 if self._split == "train" else 132,
            n_classes=4,
            first_and_last_labels=("Benign", "Normal"),
        )

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

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        for resource in self._resources:
            if os.path.isdir(self.dataset_path):
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

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        split_index_ranges = {
            "train": self._train_index_ranges,
            "val": self._val_index_ranges,
            None: [(0, 400)],
        }
        index_ranges = split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")

        return _utils.ranges_to_indices(index_ranges)
