"""FLARE22 dataset."""

import glob
import os
import re
from typing import Any, Callable, Literal

from torchvision import tv_tensors
from torchvision.datasets import utils as torch_utils
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.datasets import _utils as _data_utils
from eva.vision.data.datasets import _validators, structs
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.datasets.vision import VisionDataset


class FLARE22(VisionDataset[eva_tv_tensors.Volume, tv_tensors.Mask]):
    """FLARE22 asset with CT scans and segmentation masks.

    The FLARE 2022 Challenge focuses on semi-supervised learning for abdominal organ
    segmentation, using 50 labeled and 2000 unlabeled CT scans. Participants segment
    13 organs, with evaluation based on accuracy (DSC, NSD) and efficiency (GPU/CPU usage).
    Compared to FLARE 2021, the dataset is larger, includes more organs, and emphasizes
    resource-aware metrics.

    Dataset split was adapted from the VoCo repository:
    https://github.com/Luffy03/VoCo/blob/main/Finetune/Flare22/dataset/dataset.json

    More information can be found in:
    - https://flare22.grand-challenge.org/
    - https://zenodo.org/records/7860267
    """

    _split_index_ranges = {
        "train": [(11, 50)],
        "val": [(0, 11)],
        None: [(0, 50)],
    }
    """Sample indices for the dataset splits."""

    _resources: list[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="FLARE22Train.zip",
            url="https://zenodo.org/records/7860267/files/FLARE22Train.zip",
        ),
    ]
    """Dataset resources."""

    _license: str = (
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International "
        "(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)"
    )
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None = None,
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Path to the dataset root directory.
            split: Dataset split to use ('train' or 'val').
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
            "Liver",
            "Esophagus",
            "Stomach",
            "Duodenum",
            "Left Kidney",
            "Right kidney",
            "Spleen",
            "Pancreas",
            "Aorta",
            "Inferior vena cava",
            "Right adrenal gland",
            "Left adrenal gland",
            "Gallbladder",
        ]

    @property
    @override
    def class_to_idx(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @property
    def _dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        return os.path.join(self._root, "FLARE22Train")

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

        if len(self._samples) < len(self._indices):
            raise OSError(f"Dataset is missing {len(self._indices) - len(self._samples)} files.")

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
            match = re.search(r"FLARE22_Tr_(\d+)", filename)
            if match is None:
                raise ValueError(f"Filename '{filename}' is not valid.")

            return int(match.group(1))

        volume_files_pattern = os.path.join(self._root, "FLARE22Train", "images", "*.nii.gz")
        volume_filenames = glob.glob(volume_files_pattern, recursive=True)
        volume_ids = {filename_id(filename): filename for filename in volume_filenames}

        segmentation_files_pattern = os.path.join(self._root, "FLARE22Train", "labels", "*.nii.gz")
        segmentation_filenames = glob.glob(segmentation_files_pattern, recursive=True)
        segmentation_ids = {filename_id(filename): filename for filename in segmentation_filenames}

        return [
            (volume_ids[file_id], segmentation_ids[file_id])
            for file_id in sorted(volume_ids.keys() & segmentation_ids.keys())
        ]

    def _make_indices(self) -> list[int]:
        """Builds the dataset indices for the specified split."""
        index_ranges = self._split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")

        return _data_utils.ranges_to_indices(index_ranges)

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        for resource in self._resources:
            if os.path.isdir(self._dataset_path):
                continue

            self._print_license()
            torch_utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
