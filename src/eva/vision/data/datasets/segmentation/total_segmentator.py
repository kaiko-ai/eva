"""TotalSegmentator 2D segmentation dataset class."""

import functools
import os
from glob import glob
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import _utils, _validators, structs
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import convert, io


class TotalSegmentator2D(base.ImageSegmentation):
    """TotalSegmentator 2D segmentation dataset."""

    _expected_dataset_lengths: Dict[str, int] = {
        "train_small": 29892,
        "val_small": 6480,
    }
    """Dataset version and split to the expected size."""

    _sample_every_n_slices: int | None = None
    """The amount of slices to sub-sample per 3D CT scan image."""

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
        version: Literal["small", "full"] | None = "small",
        download: bool = False,
        as_uint8: bool = True,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If `None`, the entire dataset is used.
            version: The version of the dataset to initialize. If `None`, it will
                use the files located at root as is and wont perform any checks.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
            as_uint8: Whether to convert and return the images as a 8-bit.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._version = version
        self._download = download
        self._as_uint8 = as_uint8

        self._samples_dirs: List[str] = []
        self._indices: List[Tuple[int, int]] = []

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
        sample_idx, _ = self._indices[index]
        sample_dir = self._samples_dirs[sample_idx]
        return os.path.join(sample_dir, "ct.nii.gz")

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def configure(self) -> None:
        self._samples_dirs = self._fetch_samples_dirs()
        self._indices = self._create_indices()

    @override
    def validate(self) -> None:
        if self._version is None:
            return

        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(f"{self._split}_{self._version}", 0),
            n_classes=117,
            first_and_last_labels=("adrenal_gland_left", "vertebrae_T9"),
        )

    @override
    def __len__(self) -> int:
        return len(self._indices)

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        sample_index, slice_index = self._indices[index]
        image_path = self._get_image_path(sample_index)
        image_array = io.read_nifti_slice(image_path, slice_index)
        if self._as_uint8:
            image_array = convert.to_8bit(image_array)
        image_rgb_array = image_array.repeat(3, axis=2)
        return tv_tensors.Image(image_rgb_array.transpose(2, 0, 1))

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        sample_index, slice_index = self._indices[index]
        masks_dir = self._get_masks_dir(sample_index)
        mask_paths = (os.path.join(masks_dir, label + ".nii.gz") for label in self.classes)
        one_hot_encoded = np.concatenate(
            [io.read_nifti_slice(path, slice_index) for path in mask_paths],
            axis=2,
        )
        background_mask = one_hot_encoded.sum(axis=2, keepdims=True) == 0
        one_hot_encoded_with_bg = np.concatenate([background_mask, one_hot_encoded], axis=2)
        segmentation_label = np.argmax(one_hot_encoded_with_bg, axis=2)
        return tv_tensors.Mask(segmentation_label)

    def _get_image_path(self, sample_index: int) -> str:
        """Returns the corresponding image path."""
        sample_dir = self._samples_dirs[sample_index]
        return os.path.join(self._root, sample_dir, "ct.nii.gz")

    def _get_masks_dir(self, sample_index: int) -> str:
        """Returns the directory of the corresponding masks."""
        sample_dir = self._samples_dirs[sample_index]
        return os.path.join(self._root, sample_dir, "segmentations")

    def _get_number_of_slices_per_sample(self, sample_index: int) -> int:
        """Returns the total amount of slices of a sample."""
        image_path = self._get_image_path(sample_index)
        return io.fetch_total_nifti_slices(image_path)

    def _fetch_samples_dirs(self) -> List[str]:
        """Returns the name of all the samples of all the splits of the dataset."""
        sample_filenames = [
            filename
            for filename in os.listdir(self._root)
            if os.path.isdir(os.path.join(self._root, filename))
        ]
        return sorted(sample_filenames)

    def _get_split_indices(self) -> List[int]:
        """Returns the samples indices that corresponding the dataset split and version."""
        key = f"{self._split}_{self._version}"
        match key:
            case "train_small":
                index_ranges = [(0, 83)]
            case "val_small":
                index_ranges = [(83, 102)]
            case _:
                index_ranges = [(0, len(self._samples_dirs))]

        return _utils.ranges_to_indices(index_ranges)

    def _create_indices(self) -> List[Tuple[int, int]]:
        """Builds the dataset indices for the specified split.

        Returns:
            A list of tuples, where the first value indicates the
            sample index which the second its corresponding slice
            index.
        """
        indices = [
            (sample_idx, slide_idx)
            for sample_idx in self._get_split_indices()
            for slide_idx in range(self._get_number_of_slices_per_sample(sample_idx))
            if slide_idx % (self._sample_every_n_slices or 1) == 0
        ]
        return indices

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        dataset_resources = {
            "small": self._resources_small,
            "full": self._resources_full,
        }
        resources = dataset_resources.get(self._version or "")
        if resources is None:
            raise ValueError(
                f"Can't download data version '{self._version}'. Use 'small' or 'full'."
            )

        for resource in resources:
            if os.path.isdir(self._root):
                continue

            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )
