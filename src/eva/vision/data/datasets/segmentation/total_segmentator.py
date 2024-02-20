"""TotalSegmentator 2D segmentation dataset class."""

import functools
import os
import random
from glob import glob
from typing import Callable, Dict, List, Literal, Tuple

import cv2
import numpy as np
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import _utils, structs
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io


class TotalSegmentator2D(base.ImageSegmentation):
    """TotalSegmentator 2D segmentation dataset."""

    _train_index_ranges: List[Tuple[int, int]] = [(0, 83)]
    """Train range indices."""

    _val_index_ranges: List[Tuple[int, int]] = [(83, 103)]
    """Validation range indices."""

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
        image_target_transforms: Callable | None = None,
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
            image_target_transforms: A function/transforms that takes in an
                image and a label and returns the transformed versions of both.
                This transform happens after the `image_transforms` and
                `target_transforms`.
        """
        super().__init__(
            image_transforms=image_transforms,
            target_transforms=target_transforms,
            image_target_transforms=image_target_transforms,
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
        first_sample_labels = os.path.join(
            self._root, self._samples_dirs[0], "segmentations", "*.nii.gz"
        )
        return sorted(os.path.basename(path).split(".")[0] for path in glob(first_sample_labels))

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

    @override
    def setup(self) -> None:
        self._samples_dirs = self._fetch_samples_dirs()
        self._indices = self._create_indices()

    @override
    def load_image_and_mask(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_3D = self._load_image(index)
        image_2D, slice_idx = self._extract_image_slice(image_3D)
        mask = self._load_mask(index, slice_idx)
        return image_2D, mask

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _load_image(self, index: int) -> np.ndarray:
        """Loads and returns the `index`'th image sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The 3D grayscale image (height, width, slices) as a numpy array.
        """
        sample_dir = self._samples_dirs[self._indices[index]]
        image_path = os.path.join(self._root, sample_dir, "ct.nii.gz")
        return io.read_nifti(image_path)

    def _extract_image_slice(self, image_3D: np.ndarray) -> Tuple[np.ndarray, int]:
        """Randomly extracts one 2D image from a 3D along with its slice index.

        Args:
            image_3D: The grayscale 3D image (height, weight, n_slices).

        Returns:
            A 2D RGB image (height, width, 3) along with its corresponding 3D slice index.
        """
        slice_idx = random.randrange(image_3D.shape[2])  # nosec
        image_rgb = cv2.cvtColor(image_3D[:, :, slice_idx], cv2.COLOR_GRAY2RGB)
        return image_rgb, slice_idx

    def _load_mask(self, index: int, slice_index: int = 0) -> np.ndarray:
        """Returns the `index`'th target mask sample.

        Args:
            index: The index of the data sample target mask to load.
            slice_index: The slice index to fetch.

        Returns:
            The sample mask as a stack of binary mask arrays (label, height, width).
        """
        sample_dir = self._samples_dirs[self._indices[index]]
        masks_dir = os.path.join(self._root, sample_dir, "segmentations")
        mask_paths = (os.path.join(masks_dir, label + ".nii.gz") for label in self.classes)
        return np.stack([io.read_nifti(path, slice_index) for path in mask_paths])

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
        resources = dataset_resources.get(self._split)
        if resources is None:
            raise ValueError("Invalid data split. Use 'small' or 'full'.")

        for resource in resources:
            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )
