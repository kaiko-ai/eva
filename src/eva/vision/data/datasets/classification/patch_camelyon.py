"""PatchCamelyon dataset."""

import os
from typing import Callable, List, Literal

import h5py
import numpy as np
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import structs
from eva.vision.data.datasets.classification import base

URL_TEMPLATE = "https://zenodo.org/records/2546921/files/{filename}.gz?download=1"
"""PatchCamelyon URL files templates."""


class PatchCamelyon(base.ImageClassification):
    """Dataset class for PatchCamelyon images and corresponding targets."""

    train_resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_train_x.h5",
            url=URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_train_x.h5"),
            md5="01844da899645b4d6f84946d417ba453",
        ),
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_train_y.h5",
            url=URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_train_y.h5"),
            md5="0781386bf6c2fb62d58ff18891466aca",
        ),
    ]
    """Train resources."""

    valid_resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_valid_x.h5",
            url=URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_valid_x.h5"),
            md5="81cf9680f1724c40673f10dc88e909b1",
        ),
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_valid_y.h5",
            url=URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_valid_y.h5"),
            md5="94d8aacc249253159ce2a2e78a86e658",
        ),
    ]
    """Validation resources."""

    test_resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_test_x.h5",
            url=URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_test_x.h5"),
            md5="2614b2e6717d6356be141d9d6dbfcb7e",
        ),
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_test_y.h5",
            url=URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_test_y.h5"),
            md5="11ed647efe9fe457a4eb45df1dba19ba",
        ),
    ]
    """Test resources."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test"],
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
        super().__init__(
            image_transforms=image_transforms,
            target_transforms=target_transforms,
        )

        self._root = root
        self._split = split
        self._download = download

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def load_image(self, index: int) -> np.ndarray:
        return self._load_from_h5("x", index)

    @override
    def load_target(self, index: int) -> np.ndarray:
        target = self._load_from_h5("y", index).squeeze()
        return np.asarray(target, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return self._fetch_dataset_length()

    def _download_dataset(self) -> None:
        """Downloads the PatchCamelyon dataset."""
        dataset_resources = {
            "train": self.train_resources,
            "valid": self.valid_resources,
            "test": self.test_resources,
        }
        resources = dataset_resources.get(self._split)
        if resources is None:
            raise ValueError("Invalid data split. Use 'train', 'valid', or 'test'.")

        for resource in resources:
            file_path = os.path.join(self._root, resource.filename)
            if utils.check_integrity(file_path, resource.md5):
                continue

            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename + ".gz",
                remove_finished=True,
            )

    def _load_from_h5(
        self,
        data_key: Literal["x", "y"],
        index: int | None = None,
    ) -> np.ndarray:
        """Load data or targets from an HDF5 file.

        Args:
            data_key: Specify whether to load 'x' or 'y'.
            index: Optional parameter to load data/targets at a specific index.
                If `None`, the entire data/targets array is returned.

        Returns:
            A array containing the specified data.
        """
        h5_file = self._h5_file(data_key)
        with h5py.File(h5_file, "r") as file:
            data = file[data_key]
            return data[:] if index is None else data[index]  # type: ignore

    def _fetch_dataset_length(self) -> int:
        """Fetches the dataset split length from its HDF5 file."""
        h5_file = self._h5_file("y")
        with h5py.File(h5_file, "r") as file:
            data = file["y"]
            return len(data)  # type: ignore

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
