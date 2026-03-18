"""KiTS 2023 dataset."""

import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Literal

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from torchvision import tv_tensors
from tqdm import tqdm
from typing_extensions import override

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.datasets import _utils as _data_utils
from eva.vision.data.datasets import _validators
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.utils import ops


class KiTS23(VisionDataset[eva_tv_tensors.Volume, tv_tensors.Mask]):
    """The 2023 Kidney and Kidney Tumor Segmentation Challenge (KiTS23).

    More info:
      - The 2023 Kidney and Kidney Tumor Segmentation Challenge
        https://kits-challenge.org/kits23/
      - Dataset Split
        https://github.com/Luffy03/Large-Scale-Medical/blob/main/Downstream/monai/KiTs/dataset_kits.json
    """

    _train_index_ranges: list[tuple[int, int]] = [
        (0, 5),
        (6, 9),
        (10, 15),
        (17, 25),
        (26, 29),
        (30, 31),
        (33, 48),
        (49, 50),
        (51, 53),
        (54, 57),
        (58, 60),
        (62, 65),
        (66, 67),
        (69, 76),
        (77, 81),
        (82, 96),
        (97, 100),
        (101, 104),
        (105, 108),
        (109, 113),
        (114, 300),
    ]
    """Train range indices."""

    _val_index_ranges: list[tuple[int, int]] = [
        (5, 6),
        (9, 10),
        (15, 17),
        (25, 26),
        (29, 30),
        (31, 33),
        (48, 49),
        (50, 51),
        (53, 54),
        (57, 58),
        (60, 62),
        (65, 66),
        (67, 69),
        (76, 77),
        (81, 82),
        (96, 97),
        (100, 101),
        (104, 105),
        (108, 109),
        (113, 114),
    ]
    """Validation range indices."""

    _license: str = "MIT (https://opensource.org/license/mit)"
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

        self._samples: list[str]
        self._indices: list[int]

    @property
    @override
    def classes(self) -> list[str]:
        return ["background", "kidney", "tumor", "cyst"]

    @property
    @override
    def class_to_idx(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        ct_scan_file = self._ct_scan_filename(self._samples[self._indices[index]])
        return os.path.relpath(ct_scan_file, self._root)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def configure(self) -> None:
        self._samples = self._fetch_sample_dirs()
        self._indices = self._make_indices()

    @override
    def validate(self) -> None:
        def _valid_sample(index: int) -> bool:
            """Indicates if the sample files exist and are reachable."""
            sample_dir = self._samples[self._indices[index]]
            ct_scan_file = self._ct_scan_filename(sample_dir)
            segmentation_file = self._segmentation_filename(sample_dir)
            return os.path.isfile(ct_scan_file) and os.path.isfile(segmentation_file)

        if len(self._samples) != 489:
            raise OSError(f"Dataset is incomplete; missing {489 - len(self._samples)} samples.")

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
        sample_dir = self._samples[self._indices[index]]
        ct_scan_file = self._ct_scan_filename(sample_dir)
        return _utils.load_volume_tensor(ct_scan_file)

    @override
    def load_target(self, index: int) -> tv_tensors.Mask:
        """Loads the segmentation mask for a given sample.

        Args:
            index: The index of the desired sample.

        Returns:
            Tensor representing the segmentation mask of shape `[T, C, H, W]`.
        """
        sample_dir = self._samples[self._indices[index]]
        ct_scan_file = self._ct_scan_filename(sample_dir)
        segmentation_file = self._segmentation_filename(sample_dir)
        return _utils.load_mask_tensor(segmentation_file, ct_scan_file)

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

    def _fetch_sample_dirs(self) -> list[str]:
        """Retrieves the file paths for the sample directories.

        Returns:
            The a list of sample directory paths to the CT volumes and segmentation.
        """
        sample_dir_pattern = os.path.join(self._root, "case_**")
        sample_dirs = glob.glob(sample_dir_pattern, recursive=True)
        return ops.numeric_sort(sample_dirs)

    def _ct_scan_filename(self, sample_dir: str) -> str:
        """Returns the full path to the volume CT scan file."""
        return os.path.join(sample_dir, "imaging.nii.gz")

    def _segmentation_filename(self, sample_dir: str) -> str:
        """Returns the full path to the segmentation file."""
        return os.path.join(sample_dir, "segmentation.nii.gz")

    def _make_indices(self) -> list[int]:
        """Builds the dataset indices for the specified split."""
        split_index_ranges = {
            "train": self._train_index_ranges,
            "val": self._val_index_ranges,
            None: [(0, 300)],
        }
        index_ranges = split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")

        return _data_utils.ranges_to_indices(index_ranges)

    def _download_dataset(self) -> None:
        """Downloads the KiTS23 dataset with high concurrency and atomicity."""
        # Define ranges cleanly
        case_numbers = list(range(300)) + list(range(400, 589))
        root_path = Path(self._root)
        root_path.mkdir(parents=True, exist_ok=True)

        # Filter missing cases using Path logic
        missing_cases = [
            n for n in case_numbers if not (root_path / f"case_{n:05d}" / "imaging.nii.gz").exists()
        ]

        if not missing_cases:
            logger.info("Dataset already complete.")
            return

        logger.info(f"Downloading {len(missing_cases)} cases using 8 workers...")

        # Use ThreadPoolExecutor for IO-bound task (downloads)
        with tqdm(total=len(missing_cases), desc="KiTS23 Download") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                # Schedule all downloads
                futures = {
                    executor.submit(download_kits23_case, str(root_path), n): n
                    for n in missing_cases
                }

                for future in as_completed(futures):
                    case_n = futures[future]
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Case {case_n} failed after retries: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def download_resource(url: str, dest: Path, chunk_size: int = 1024 * 1024):
    """Generic, resumable, and atomic download utility."""
    tmp_path = dest.with_suffix(dest.suffix + ".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)

    resume_header = {}
    mode = "wb"
    if tmp_path.exists():
        resume_header = {"Range": f"bytes={tmp_path.stat().st_size}-"}
        mode = "ab"

    with requests.get(url, stream=True, headers=resume_header, timeout=30) as r:
        r.raise_for_status()
        with open(tmp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    tmp_path.replace(dest)  # Atomic move


def download_kits23_case(root: str, case_num: int):
    """Orchestrates the download for a specific case."""
    BASE_URL = "https://kits19.sfo2.digitaloceanspaces.com"
    RAW_URL = "https://raw.githubusercontent.com/neheller/kits23/main/dataset"

    case_id = f"case_{case_num:05d}"
    case_dir = Path(root) / case_id

    files_to_get = {
        f"{BASE_URL}/master_{case_num:05d}.nii.gz": case_dir / "imaging.nii.gz",
        f"{RAW_URL}/{case_id}/segmentation.nii.gz": case_dir / "segmentation.nii.gz",
    }

    try:
        for url, dest in files_to_get.items():
            if not dest.exists():
                download_resource(url, dest)
    except Exception as e:
        logger.error(f"Permanent failure for {case_id}: {e}")
        raise
