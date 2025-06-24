"""BTCV dataset."""

import glob
import os
import re
from typing import Any, Callable, Dict, List, Literal, Tuple

import huggingface_hub
from torchvision import tv_tensors
from torchvision.datasets import utils as data_utils
from typing_extensions import override

from eva.core.utils import requirements
from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.datasets import _utils as _data_utils
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.datasets.vision import VisionDataset


class BTCV(VisionDataset[eva_tv_tensors.Volume, tv_tensors.Mask]):
    """Beyond the Cranial Vault (BTCV) Abdomen dataset.

    The BTCV dataset comprises abdominal CT acquired at the Vanderbilt
    University Medical Center from metastatic liver cancer patients or
    post-operative ventral hernia patients. The dataset contains one
    background class and thirteen organ classes.

    More info:
      - Multi-organ Abdominal CT Reference Standard Segmentations
        https://zenodo.org/records/1169361
      - Dataset Split
        https://github.com/Luffy03/Large-Scale-Medical/blob/main/Downstream/monai/BTCV/dataset/dataset_0.json
    """

    _split_index_ranges = {
        "train": [(0, 24)],
        "val": [(24, 30)],
        None: [(0, 30)],
    }
    """Sample indices for the dataset splits."""

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

        self._samples: List[Tuple[str, str]]
        self._indices: List[int]

    @property
    @override
    def classes(self) -> List[str]:
        return [
            "background",
            "spleen",
            "right_kidney",
            "left_kidney",
            "gallbladder",
            "esophagus",
            "liver",
            "stomach",
            "aorta",
            "inferior_vena_cava",
            "portal_and_splenic_vein",
            "pancreas",
            "right_adrenal_gland",
            "left_adrenal_gland",
        ]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

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
        requirements.check_dependencies(requirements={"torch": "2.5.1", "torchvision": "0.20.1"})

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

        subdir = os.path.join(self._root, "BTCV")
        root = subdir if os.path.isdir(subdir) else self._root

        volume_files_pattern = os.path.join(root, "imagesTr", "*.nii.gz")
        volume_filenames = glob.glob(volume_files_pattern)
        volume_ids = {filename_id(filename): filename for filename in volume_filenames}

        segmentation_files_pattern = os.path.join(root, "labelsTr", "*.nii.gz")
        segmentation_filenames = glob.glob(segmentation_files_pattern)
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
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Huggingface token required, please set the HF_TOKEN env variable.")

        huggingface_hub.snapshot_download(
            "Luffy503/VoCo_Downstream",
            repo_type="dataset",
            token=hf_token,
            local_dir=self._root,
            ignore_patterns=[".git*"],
            allow_patterns=["BTCV.zip"],
        )

        zip_path = os.path.join(self._root, "BTCV.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"BTCV.zip not found in {self._root}, something with the download went wrong."
            )

        data_utils.extract_archive(zip_path, self._root, remove_finished=True)
