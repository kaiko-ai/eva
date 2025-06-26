"""LiTS17 dataset."""

import glob
import os
import re
from typing import Any, Callable, Dict, List, Literal, Tuple

from torchvision import tv_tensors
from typing_extensions import override

from eva.core.utils import requirements
from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.datasets import _utils as _data_utils
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.datasets.vision import VisionDataset


class LiTS17(VisionDataset[eva_tv_tensors.Volume, tv_tensors.Mask]):
    """LiTS17 - Liver Tumor Segmentation Challenge 2017.

    More info:
      - The Liver Tumor Segmentation Benchmark (LiTS)
        https://arxiv.org/pdf/1901.04056
      - Dataset Split
        https://github.com/Luffy03/Large-Scale-Medical/blob/main/Downstream/monai/LiTs/dataset_lits.json
      - Data needs to be manually downloaded from:
        https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE
    """

    _train_index_ranges: List[Tuple[int, int]] = [
        (0, 2),
        (4, 14),
        (15, 16),
        (18, 48),
        (50, 51),
        (52, 57),
        (58, 65),
        (66, 67),
        (71, 74),
        (75, 81),
        (82, 85),
        (86, 92),
        (93, 99),
        (102, 103),
        (104, 116),
        (117, 123),
        (124, 126),
        (127, 131),
    ]
    """Train range indices."""

    _val_index_ranges: List[Tuple[int, int]] = [
        (2, 4),
        (14, 15),
        (16, 18),
        (48, 50),
        (51, 52),
        (57, 58),
        (65, 66),
        (67, 68),
        (70, 71),
        (74, 75),
        (81, 82),
        (85, 86),
        (92, 93),
        (99, 102),
        (103, 104),
        (116, 117),
        (123, 124),
        (126, 127),
    ]
    """Validation range indices."""

    _split_index_ranges = {
        "train": _train_index_ranges,
        "val": _val_index_ranges,
        None: [(0, 128)],
    }
    """Sample indices for the dataset splits."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Path to the dataset root directory.
            split: Dataset split to use ('train' or 'val').
                If None, it uses the full dataset.
            transforms: A callable object for applying data transformations.
                If None, no transformations are applied.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._transforms = transforms

        self._samples: Dict[int, Tuple[str, str]]
        self._indices: List[int]

    @property
    @override
    def classes(self) -> List[str]:
        return ["background", "liver", "tumor"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        return os.path.relpath(self._samples[self._indices[index]][0], self._root)

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
    ) -> Tuple[eva_tv_tensors.Volume, tv_tensors.Mask]:
        """Applies transformations to the provided data.

        Args:
            ct_scan: The CT volume tensor.
            mask: The segmentation mask tensor.

        Returns:
            A tuple containing the transformed CT and mask tensors.
        """
        return self._transforms(ct_scan, mask) if self._transforms else (ct_scan, mask)

    def _find_samples(self) -> Dict[int, Tuple[str, str]]:
        """Retrieves the file paths for the CT volumes and segmentation.

        Returns:
            The a dictionary mapping file IDs to tuples of volume and segmentation file paths.
        """

        def filename_id(filename: str) -> int:
            matches = re.match(r".*(?:\D|^)(\d+)", filename)
            if matches is None:
                raise ValueError(f"Filename '{filename}' is not valid.")

            return int(matches.group(1))

        volume_files_pattern = os.path.join(self._root, "**", "volume-*.nii")
        volume_filenames = glob.glob(volume_files_pattern, recursive=True)
        volume_ids = {filename_id(filename): filename for filename in volume_filenames}

        segmentation_files_pattern = os.path.join(self._root, "**", "segmentation-*.nii")
        segmentation_filenames = glob.glob(segmentation_files_pattern, recursive=True)
        segmentation_ids = {filename_id(filename): filename for filename in segmentation_filenames}

        return {
            file_id: (volume_ids[file_id], segmentation_ids[file_id])
            for file_id in sorted(volume_ids.keys() & segmentation_ids.keys())
        }

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        index_ranges = self._split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")

        return _data_utils.ranges_to_indices(index_ranges)
