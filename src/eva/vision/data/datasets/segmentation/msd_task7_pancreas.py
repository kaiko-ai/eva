"""Dataset for Task 7 (pancreas tumor) from the Medical Segmentation Decathlon (MSD)."""

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
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.datasets.segmentation.metadata import _msd_task7_pancreas
from eva.vision.data.datasets.vision import VisionDataset


class MSDTask7Pancreas(VisionDataset[eva_tv_tensors.Volume, tv_tensors.Mask]):
    """Task 7 (pancreas tumor) of the Medical Segmentation Decathlon (MSD).

    The data set consists of 420 portal-venous phase CT scans of patients undergoing
    resection of pancreatic masses. The corresponding target ROIs were the pancreatic
    parenchyma and pancreatic mass (cyst or tumor). This data set was selected due to
    label unbalance between large (background), medium (pancreas) and small (tumor)
    structures. The data was acquired in the Memorial Sloan Kettering Cancer
    Center, New York, US.

    More info:
      - Paper: https://www.nature.com/articles/s41467-022-30695-9
      - Dataset source: https://huggingface.co/datasets/Luffy503/VoCo_Downstream
    """

    _train_ids = _msd_task7_pancreas.train_ids
    """File indices of the training set."""

    _val_ids = _msd_task7_pancreas.val_ids
    """File indices of the validation set."""

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
        super().__init__()

        self._root = root
        self._split = split
        self._download = download
        self._transforms = transforms

        self._samples: Dict[int, Tuple[str, str]]
        self._indices: List[int]

    @property
    @override
    def classes(self) -> List[str]:
        return [
            "background",
            "pancreas",
            "cancer",
        ]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        return os.path.relpath(self._samples[self._indices[index]][0], self._root)

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
            The a dictionary mapping the file id to the volume and segmentation paths.
        """

        def filename_id_volume(filename: str) -> int:
            matches = re.match(r".*(\d{3})_\d{4}.*", filename)
            if matches is None:
                raise ValueError(f"Filename '{filename}' is not valid.")
            return int(matches.group(1))

        def filename_id_segmentation(filename: str) -> int:
            matches = re.match(r".*(\d{3}).*", filename)
            if matches is None:
                raise ValueError(f"Filename '{filename}' is not valid.")
            return int(matches.group(1))

        optional_subdir = os.path.join(self._root, "Dataset007_Pancreas")
        search_dir = optional_subdir if os.path.isdir(optional_subdir) else self._root

        volume_files_pattern = os.path.join(search_dir, "imagesTr", "*.nii.gz")
        volume_filenames = glob.glob(volume_files_pattern)
        volume_ids = {filename_id_volume(filename): filename for filename in volume_filenames}

        segmentation_files_pattern = os.path.join(search_dir, "labelsTr", "*.nii.gz")
        segmentation_filenames = glob.glob(segmentation_files_pattern)
        segmentation_ids = {
            filename_id_segmentation(filename): filename for filename in segmentation_filenames
        }

        return {
            file_id: (volume_ids[file_id], segmentation_ids[file_id])
            for file_id in volume_ids.keys()
        }

    def _make_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        file_ids = []
        match self._split:
            case "train":
                file_ids = self._train_ids
            case "val":
                file_ids = self._val_ids
            case None:
                file_ids = self._train_ids + self._val_ids
            case _:
                raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")

        return file_ids

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
            allow_patterns=["**/Dataset007_Pancreas.zip"],
        )

        zip_path = os.path.join(self._root, "MSD_Decathlon/Dataset007_Pancreas.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"MSD_Decathlon/Dataset007_Pancreas.zip not found in {self._root}, "
                "something with the download went wrong."
            )

        data_utils.extract_archive(zip_path, self._root, remove_finished=True)
