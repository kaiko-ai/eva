"""BreaKHis dataset class."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Set

import torch
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import _validators, structs, vision
from eva.vision.utils import io


class BreaKHis(vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for BreaKHis images and corresponding targets."""

    _resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="BreaKHis_v1.tar.gz",
            url="http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
        ),
    ]
    """Dataset resources."""

    _val_patient_ids: Set[str] = {
        "18842D",
        "19979",
        "15275",
        "15792",
        "16875",
        "3909",
        "5287",
        "16716",
        "2773",
        "5695",
        "16184CD",
        "23060CD",
        "21998CD",
        "21998EF",
    }
    """Patient IDs to use for dataset splits."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 1132,
        "val": 339,
        None: 1471,
    }
    """Expected dataset lengths for the splits and complete dataset."""

    _default_magnifications = ["40X"]
    """Default magnification to use for images in train/val datasets."""

    _license: str = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None = None,
        magnifications: List[Literal["40X", "100X", "200X", "400X"]] | None = None,
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        The dataset is split into train and validation by taking into account
        the patient IDs to avoid any data leakage.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If `None`, the entire dataset is used.
            magnifications: A list of the WSI magnifications to select. By default
                only 40X images are used.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does
                not yet exist on disk.
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._download = download

        self._magnifications = magnifications or self._default_magnifications
        self._indices: List[int] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["TA", "MC", "F", "DC"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @property
    def _dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        return os.path.join(self._root, "BreaKHis_v1", "histology_slides")

    @functools.cached_property
    def _image_files(self) -> List[str]:
        """Return the list of image files in the dataset.

        Returns:
            List of image file paths.
        """
        image_files = []
        for magnification in self._magnifications:
            files_pattern = os.path.join(self._dataset_path, f"**/{magnification}", "*.png")
            image_files.extend(list(glob.glob(files_pattern, recursive=True)))
        return sorted(image_files)

    @override
    def filename(self, index: int) -> str:
        image_path = self._image_files[self._indices[index]]
        return os.path.relpath(image_path, self._dataset_path)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        _validators.check_dataset_exists(self._root, True)

    @override
    def configure(self) -> None:
        self._indices = self._make_indices()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths[self._split],
            n_classes=4,
            first_and_last_labels=("TA", "DC"),
        )

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_path = self._image_files[self._indices[index]]
        return io.read_image_as_tensor(image_path)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        class_name = self._extract_class(self._image_files[self._indices[index]])
        return torch.tensor(self.class_to_idx[class_name], dtype=torch.long)

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return {"patient_id": self._extract_patient_id(self._image_files[self._indices[index]])}

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        for resource in self._resources:
            if os.path.isdir(self._dataset_path):
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

    def _extract_patient_id(self, image_file: str) -> str:
        """Extracts the patient ID from the image file name."""
        return os.path.basename(image_file).split("-")[2]

    def _extract_class(self, file: str) -> str:
        return os.path.basename(file).split("-")[0].split("_")[-1]

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        train_indices = []
        val_indices = []

        for index, image_file in enumerate(self._image_files):
            if self._extract_class(image_file) not in self.classes:
                continue
            patient_id = self._extract_patient_id(image_file)
            if patient_id in self._val_patient_ids:
                val_indices.append(index)
            else:
                train_indices.append(index)

        split_indices = {
            "train": train_indices,
            "val": val_indices,
            None: train_indices + val_indices,
        }

        return split_indices[self._split]
