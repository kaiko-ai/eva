"""TotalSegmentator 2D segmentation dataset class."""

import functools
import os
from glob import glob
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import _utils, _validators, structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class TotalSegmentatorClassification(base.ImageClassification):
    """TotalSegmentator multi-label classification dataset."""

    _train_index_ranges: List[Tuple[int, int]] = [(0, 83)]
    """Train range indices."""

    _val_index_ranges: List[Tuple[int, int]] = [(83, 103)]
    """Validation range indices."""

    _n_slices_per_image: int = 20
    """The amount of slices to sample per 3D CT scan image."""

    _resources_full: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_v201.zip",
            url="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip",
            md5="fe250e5718e0a3b5df4c4ea9d58a62fe",
        ),
    ]
    """Resources for the full dataset version."""

    _resources_small: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_small_v201.zip",
            url="https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip",
            md5="6b5524af4b15e6ba06ef2d700c0c73e0",
        ),
    ]
    """Resources for the small dataset version."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None,
        version: Literal["small", "full"] = "small",
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.
            version: The version of the dataset to initialize.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
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
        self._version = version
        self._download = download

        self._samples_dirs: List[str] = []
        self._indices: List[int] = []

    @functools.cached_property
    @override
    def classes(self) -> List[str]:
        def get_filename(path: str) -> str:
            """Returns the filename from the full path."""
            return os.path.basename(path).split(".")[0]

        first_sample_labels = os.path.join(
            self._root, self._samples_dirs[0], "segmentations", "*.nii.gz"
        )
        return sorted(map(get_filename, glob(first_sample_labels)))

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        sample_dir = self._samples_dirs[self._indices[index]]
        return os.path.join(sample_dir, "ct.nii.gz")

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        _validators.check_dataset_exists(self._root, True)

    @override
    def configure(self) -> None:
        self._samples_dirs = self._fetch_samples_dirs()
        self._indices = self._create_indices()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=1660 if self._split == "train" else 400,
            n_classes=117,
            first_and_last_labels=("adrenal_gland_left", "vertebrae_T9"),
        )

    @override
    def __len__(self) -> int:
        return len(self._indices) * self._n_slices_per_image

    @override
    def load_image(self, index: int) -> np.ndarray:
        image_path = self._get_image_path(index)
        slice_index = self._get_sample_slice_index(index)
        image_array = io.read_nifti_slice(image_path, slice_index)
        return image_array.repeat(3, axis=2)

    @override
    def load_target(self, index: int) -> np.ndarray:
        masks = self._load_masks(index)
        targets = [1 in masks[..., mask_index] for mask_index in range(masks.shape[-1])]
        return np.asarray(targets, dtype=np.int64)

    def _load_masks(self, index: int) -> np.ndarray:
        """Returns the `index`'th target mask sample."""
        masks_dir = self._get_masks_dir(index)
        slice_index = self._get_sample_slice_index(index)
        mask_paths = (os.path.join(masks_dir, label + ".nii.gz") for label in self.classes)
        masks = [io.read_nifti_slice(path, slice_index) for path in mask_paths]
        return np.concatenate(masks, axis=-1)

    def _get_masks_dir(self, index: int) -> str:
        """Returns the directory of the corresponding masks."""
        sample_dir = self._get_sample_dir(index)
        return os.path.join(self._root, sample_dir, "segmentations")

    def _get_image_path(self, index: int) -> str:
        """Returns the corresponding image path."""
        sample_dir = self._get_sample_dir(index)
        return os.path.join(self._root, sample_dir, "ct.nii.gz")

    def _get_sample_dir(self, index: int) -> str:
        """Returns the corresponding sample directory."""
        sample_index = self._indices[index // self._n_slices_per_image]
        return self._samples_dirs[sample_index]

    def _get_sample_slice_index(self, index: int) -> int:
        """Returns the corresponding slice index."""
        image_path = self._get_image_path(index)
        total_slices = io.fetch_total_nifti_slices(image_path)
        slice_indices = np.linspace(0, total_slices - 1, num=self._n_slices_per_image, dtype=int)
        return slice_indices[index % self._n_slices_per_image]

    def _fetch_samples_dirs(self) -> List[str]:
        """Returns the name of all the samples of all the splits of the dataset."""
        sample_filenames = [
            filename
            for filename in os.listdir(self._root)
            if os.path.isdir(os.path.join(self._root, filename))
        ]
        return sorted(sample_filenames)

    def _create_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        split_index_ranges = {
            "train": self._train_index_ranges,
            "val": self._val_index_ranges,
            None: [(0, 103)],
        }
        index_ranges = split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")

        return _utils.ranges_to_indices(index_ranges)

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        dataset_resources = {
            "small": self._resources_small,
            "full": self._resources_full,
            None: (0, 103),
        }
        resources = dataset_resources.get(self._version)
        if resources is None:
            raise ValueError("Invalid data version. Use 'small' or 'full'.")

        for resource in resources:
            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )
