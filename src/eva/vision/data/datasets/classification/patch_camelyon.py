"""PatchCamelyon dataset."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import h5py
import numpy as np
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets.classification import base


class PatchCamelyon(base.ImageClassification):
    """Dataset class for PatchCamelyon images and corresponding targets."""

    column_mapping: Dict[str, Literal["x", "y"]] = {
        "data": "x",
        "targets": "y",
    }
    """The column mapping of the variables to the H5 columns."""

    train_list = [
        ("camelyonpatch_level_2_split_train_x.h5", "01844da899645b4d6f84946d417ba453"),
        ("camelyonpatch_level_2_split_train_y.h5", "0781386bf6c2fb62d58ff18891466aca"),
    ]
    valid_list = [
        ("camelyonpatch_level_2_split_valid_x.h5", "81cf9680f1724c40673f10dc88e909b1"),
        ("camelyonpatch_level_2_split_valid_y.h5", "94d8aacc249253159ce2a2e78a86e658"),
    ]
    test_list = [
        ("camelyonpatch_level_2_split_test_x.h5", "2614b2e6717d6356be141d9d6dbfcb7e"),
        ("camelyonpatch_level_2_split_test_y.h5", "11ed647efe9fe457a4eb45df1dba19ba"),
    ]
    """PatchCamelyon dataset split file lists."""

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
        super().__init__(image_transforms=image_transforms, target_transforms=target_transforms)

        self._root = root
        self._split = split
        self._download = download
        self._image_transforms = image_transforms
        self._target_transforms = target_transforms

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def load_image(self, index: int) -> np.ndarray:
        return self._load_from_h5("data", index)

    @override
    def load_target(self, index: int) -> np.ndarray:
        target = self._load_from_h5("targets", index).squeeze()
        return np.asarray(target, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return self._fetch_dataset_length()

    def _fetch_dataset_length(self) -> int:
        """Fetches the dataset split length from its HDF5 file."""
        h5_file = self._h5_file(self.column_mapping["targets"])
        with h5py.File(h5_file, "r") as file:
            data = file[self.column_mapping["targets"]]
            return len(data)  # type: ignore

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
            return data[:] if index is None else data[index]  # type: ignore

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

    @property
    def _download_list(self) -> List[Tuple[str, str]]:
        """Returns the appropriate download list based on the current data split."""
        match self._split:
            case "train":
                return self.train_list
            case "valid":
                return self.valid_list
            case "test":
                return self.test_list
            case _:
                raise ValueError("Invalid data split. Use 'train', 'val', or 'test'.")

    def _download_dataset(self) -> None:
        """Downloads the PatchCamelyon dataset."""
        for filename, md5 in self._download_list:
            file_path = os.path.join(self._root, filename)
            if utils.check_integrity(file_path, md5):
                continue

            url = self._filename_to_url(filename)
            utils.download_and_extract_archive(
                url,
                download_root=self._root,
                filename=filename + ".gz",
                remove_finished=True,
            )

    @staticmethod
    def _filename_to_url(filename: str) -> str:
        """Convert a filename to its corresponding download URL."""
        return f"https://zenodo.org/records/2546921/files/{filename}.gz?download=1"
