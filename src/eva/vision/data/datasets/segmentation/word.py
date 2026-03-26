"""WORD dataset."""

import glob
import os
import re
from typing import Any, Callable, Literal

import pyzipper
from torchvision import tv_tensors
from torchvision.datasets import utils as torch_utils
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.datasets import _validators, structs
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.datasets.vision import VisionDataset


class WORD(VisionDataset[eva_tv_tensors.Volume, tv_tensors.Mask]):
    """Whole abdominal Organs Dataset.

    WORD is a dataset for organ semantic segmentation that contains 150
    abdominal CT volumes (30,495 slices) and each volume has 16 organs
    with fine pixel-level annotations and scribble-based sparse annotation,
    which may be the largest dataset with whole abdominal organs annotation.

    More info:
      - WORD: A large scale dataset, benchmark and clinical applicable
        study for abdominal organ segmentation from CT image
        https://arxiv.org/abs/2111.02403
      - Dataset Split
        https://github.com/Luffy03/Large-Scale-Medical/blob/main/Downstream/monai/Word/dataset/dataset.json
    """

    _resources: list[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="WORD-V0.1.0.zip",
            url="https://drive.google.com/file/d/19OWCXZGrimafREhXm8O8w2HBHZTfxEgU/view?usp=sharing",
        ),
    ]
    """Dataset resources."""

    _license: str = "GNU General Public License v3.0 (https://www.gnu.org/licenses/gpl-3.0.html)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None = None,
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Path to the dataset root directory.
            split: Dataset split to use ('train', 'val', or 'test').
                If None, it uses the full dataset.
            download: Whether to download the dataset.
            transforms: A callable object for applying data transformations.
                If None, no transformations are applied.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._download = download
        self._transforms = transforms

        self._samples: list[tuple[str, str]]
        self._indices: list[int]

    @property
    @override
    def classes(self) -> list[str]:
        return [
            "background",
            "liver",
            "colon",
            "intestine",
            "adrenal",
            "rectum",
            "bladder",
            "Head_of_femur_L",
            "Head_of_femur_R",
            "spleen",
            "left_kidney",
            "right_kidney",
            "stomach",
            "gallbladder",
            "esophagus",
            "pancreas",
            "duodenum",
        ]

    @property
    @override
    def class_to_idx(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @property
    def _dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        return os.path.join(self._root, "WORD-V0.1.0")

    @override
    def filename(self, index: int) -> str:
        return os.path.basename(self._samples[self._indices[index]][0])

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def configure(self) -> None:
        self._samples = self._find_samples()
        self._indices = self._make_indices()

    @override
    def validate(self) -> None:
        def _valid_sample(index: int) -> bool:
            """Indicates if the sample files exist and are reachable."""
            volume_file, segmentation_file = self._samples[self._indices[index]]
            return os.path.isfile(volume_file) and os.path.isfile(segmentation_file)

        if len(self._samples) != 120:
            raise OSError(f"Dataset is incomplete; missing {120 - len(self._samples)} samples.")

        invalid_samples = [self._samples[i] for i in range(len(self)) if not _valid_sample(i)]
        if invalid_samples:
            raise OSError(
                f"Dataset '{self.__class__.__qualname__}' contains missing or "
                f"corrupted samples  ({len(invalid_samples)} in total). "
                f"{_validators._SUFFIX_ERROR_MESSAGE} "
                f"Examples of missing folders: {str(invalid_samples[:10])[:-1]}, ...]. "
            )

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[eva_tv_tensors.Volume, tv_tensors.Mask, dict[str, Any]]:
        volume = self.load_data(index)
        mask = self.load_target(index)
        metadata = self.load_metadata(index) or {}
        volume_tensor, mask_tensor = self._apply_transforms(volume, mask)
        return volume_tensor, mask_tensor, metadata

    @override
    def __len__(self) -> int:
        return len(self._indices)

    @override
    def load_data(self, index: int) -> eva_tv_tensors.Volume:
        """Loads the CT volume for a given sample.

        Args:
            index: The index of the desired sample.

        Returns:
            Tensor representing the CT volume of shape `[T, C, H, W]`.
        """
        ct_scan_file, _ = self._samples[self._indices[index]]
        return _utils.load_volume_tensor(ct_scan_file)

    @override
    def load_target(self, index: int) -> tv_tensors.Mask:
        """Loads the segmentation mask for a given sample.

        Args:
            index: The index of the desired sample.

        Returns:
            Tensor representing the segmentation mask of shape `[T, C, H, W]`.
        """
        ct_scan_file, mask_file = self._samples[self._indices[index]]
        return _utils.load_mask_tensor(mask_file, ct_scan_file)

    def _apply_transforms(
        self, ct_scan: eva_tv_tensors.Volume, mask: tv_tensors.Mask
    ) -> tuple[eva_tv_tensors.Volume, tv_tensors.Mask]:
        """Applies transformations to the provided data.

        Args:
            ct_scan: The CT volume tensor.
            mask: The segmentation mask tensor.

        Returns:
            A tuple containing the transformed CT and mask tensors.
        """
        return self._transforms(ct_scan, mask) if self._transforms else (ct_scan, mask)

    def _find_samples(self) -> list[tuple[str, str]]:
        """Retrieves the file paths for the CT volumes and segmentation.

        Returns:
            The a list of file path to the CT volumes and segmentation.
        """

        def filename_id(filename: str) -> int:
            matches = re.match(r".*(?:\D|^)(\d+)", filename)
            if matches is None:
                raise ValueError(f"Filename '{filename}' is not valid.")
            return int(matches.group(1))

        volume_train_files_pattern = os.path.join(self._root, "WORD-V0.1.0", "imagesTr", "*.nii.gz")
        volume_val_files_pattern = os.path.join(self._root, "WORD-V0.1.0", "imagesVal", "*.nii.gz")
        volume_test_files_pattern = os.path.join(self._root, "WORD-V0.1.0", "imagesTs", "*.nii.gz")
        volume_filenames = (
            glob.glob(volume_train_files_pattern)
            + glob.glob(volume_val_files_pattern)
            + glob.glob(volume_test_files_pattern)
        )
        volume_ids = {filename_id(filename): filename for filename in volume_filenames}

        segmentation_train_files_pattern = os.path.join(
            self._root, "WORD-V0.1.0", "labelsTr", "*.nii.gz"
        )
        segmentation_val_files_pattern = os.path.join(
            self._root, "WORD-V0.1.0", "labelsVal", "*.nii.gz"
        )
        segmentation_test_files_pattern = os.path.join(
            self._root, "WORD-V0.1.0", "labelsTs", "*.nii.gz"
        )
        segmentation_filenames = (
            glob.glob(segmentation_train_files_pattern)
            + glob.glob(segmentation_val_files_pattern)
            + glob.glob(segmentation_test_files_pattern)
        )
        segmentation_ids = {filename_id(filename): filename for filename in segmentation_filenames}

        return [
            (volume_ids[file_id], segmentation_ids[file_id])
            for file_id in sorted(volume_ids.keys() & segmentation_ids.keys())
        ]

    def _make_indices(self) -> list[int]:
        """Builds the dataset indices for the specified split."""
        train_indices, val_indices, test_indices = [], [], []

        for index, (volume_file, _) in enumerate(self._samples):
            if "imagesTr" in volume_file:
                train_indices.append(index)
            elif "imagesVal" in volume_file:
                val_indices.append(index)
            elif "imagesTs" in volume_file:
                test_indices.append(index)
            else:
                raise ValueError(f"Invalid file path '{volume_file}'.")

        split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            None: train_indices + val_indices + test_indices,
        }
        indices = split_indices.get(self._split)
        if indices is None:
            raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")
        return indices

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        for resource in self._resources:
            if os.path.isdir(self._dataset_path):
                continue

            self._print_license()
            torch_utils.download_url(
                resource.url,
                root=self._root,
                filename=resource.filename,
            )

            archive_path = os.path.join(self._root, "WORD-V0.1.0.zip")
            with pyzipper.AESZipFile(archive_path) as zf:
                zf.extractall(path=self._root, pwd=b"word@uestc")

            os.remove(archive_path)

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
