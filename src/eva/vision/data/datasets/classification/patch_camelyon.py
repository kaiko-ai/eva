"""PatchCamelyon dataset."""

import os
from typing import Callable, Dict, List, Literal

import h5py
import torch
from torchvision import tv_tensors
from torchvision.datasets import utils
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, structs, vision

_URL_TEMPLATE = "https://zenodo.org/records/2546921/files/{filename}.gz?download=1"
"""PatchCamelyon URL files templates."""


class PatchCamelyon(vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for PatchCamelyon images and corresponding targets."""

    _train_resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_train_x.h5",
            url=_URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_train_x.h5"),
            md5="01844da899645b4d6f84946d417ba453",
        ),
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_train_y.h5",
            url=_URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_train_y.h5"),
            md5="0781386bf6c2fb62d58ff18891466aca",
        ),
    ]
    """Train resources."""

    _val_resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_valid_x.h5",
            url=_URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_valid_x.h5"),
            md5="81cf9680f1724c40673f10dc88e909b1",
        ),
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_valid_y.h5",
            url=_URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_valid_y.h5"),
            md5="94d8aacc249253159ce2a2e78a86e658",
        ),
    ]
    """Validation resources."""

    _test_resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_test_x.h5",
            url=_URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_test_x.h5"),
            md5="2614b2e6717d6356be141d9d6dbfcb7e",
        ),
        structs.DownloadResource(
            filename="camelyonpatch_level_2_split_test_y.h5",
            url=_URL_TEMPLATE.format(filename="camelyonpatch_level_2_split_test_y.h5"),
            md5="11ed647efe9fe457a4eb45df1dba19ba",
        ),
    ]
    """Test resources."""

    _license: str = (
        "Creative Commons Zero v1.0 Universal (https://choosealicense.com/licenses/cc0-1.0/)"
    )
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: The path to the dataset root. This path should contain
                the uncompressed h5 files and the metadata.
            split: The dataset split for training, validation, or testing.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method.
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._download = download

    @property
    @override
    def classes(self) -> List[str]:
        return ["no_tumor", "tumor"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {"no_tumor": 0, "tumor": 1}

    @override
    def filename(self, index: int) -> str:
        return f"camelyonpatch_level_2_split_{self._split}_x_{index}"

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        _validators.check_dataset_exists(self._root, True)

    @override
    def validate(self) -> None:
        expected_length = {
            "train": 262144,
            "val": 32768,
            "test": 32768,
        }
        _validators.check_dataset_integrity(
            self,
            length=expected_length.get(self._split, 0),
            n_classes=2,
            first_and_last_labels=("no_tumor", "tumor"),
        )

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        return self._load_from_h5("x", index)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        target = self._load_from_h5("y", index).squeeze()
        return torch.tensor(target, dtype=torch.float32)

    @override
    def __len__(self) -> int:
        return self._fetch_dataset_length()

    def _download_dataset(self) -> None:
        """Downloads the PatchCamelyon dataset."""
        for resource in self._train_resources + self._val_resources + self._test_resources:
            file_path = os.path.join(self._root, resource.filename)
            if utils.check_integrity(file_path, resource.md5):
                continue

            self._print_license()
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
    ) -> tv_tensors.Image:
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
            image_array = data[:] if index is None else data[index]  # type: ignore
            return functional.to_image(image_array)  # type: ignore

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
        split_suffix = "valid" if self._split == "val" else self._split
        filename = f"camelyonpatch_level_2_split_{split_suffix}_{datatype}.h5"
        return os.path.join(self._root, filename)

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
